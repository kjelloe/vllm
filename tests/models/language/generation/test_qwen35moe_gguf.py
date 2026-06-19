# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end coherence tests for Qwen3.5-35B-A3B GGUF (model_type=qwen3_5_moe).

These tests require:
  - The GGUF file at $LLAMA_MODELS_DIR/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf
  - At least one CUDA GPU with 24 GiB VRAM for TP=1
  - Two CUDA GPUs for TP=2

Skip automatically when the model file is not present (intended for local
development; not run in CI without the model).

To run locally:
    .venv/bin/python -m pytest tests/models/language/generation/test_qwen35moe_gguf.py -v

    # TP=1 only:
    .venv/bin/python -m pytest tests/models/language/generation/test_qwen35moe_gguf.py -v -k tp1

    # TP=2 only (requires 2 GPUs):
    .venv/bin/python -m pytest tests/models/language/generation/test_qwen35moe_gguf.py -v -k tp2
"""
from __future__ import annotations

import os

import pytest
import torch

# Hardware rig: RTX 4090 (device 0, SM89) + RTX 3090 (device 1, SM86)
# Always set PCI bus order so device indices are stable.
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

MODEL_PATH = os.path.join(
    os.environ.get("LLAMA_MODELS_DIR", ""),
    "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf",
)
HF_CONFIG_PATH = "Qwen/Qwen3.5-35B-A3B"

requires_model = pytest.mark.skipif(
    not os.path.isfile(MODEL_PATH),
    reason=f"Model file not found: {MODEL_PATH}. Set $LLAMA_MODELS_DIR.",
)

requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="TP=2 test requires at least 2 CUDA GPUs.",
)

COHERENCE_PROMPTS = [
    ("The capital of France is", "Paris"),
    ("What is 2+2? Answer:", "4"),
    ("def fibonacci(n):", "return"),  # "if" also acceptable
]

def _common_kwargs(tp_size: int) -> dict:
    """Return LLM kwargs with correct CUDA_VISIBLE_DEVICES for the requested TP."""
    if tp_size == 1:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
        gpu_mem = 0.935
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')
        gpu_mem = 0.85
    return dict(
        tokenizer=HF_CONFIG_PATH,
        hf_config_path=HF_CONFIG_PATH,
        enforce_eager=True,
        max_model_len=256,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=tp_size,
    )


def _run_coherence_check(llm, prompts_with_expected, params):
    """Run prompts and check that expected substrings appear in outputs."""
    prompts = [p for p, _ in prompts_with_expected]
    outputs = llm.generate(prompts, params)
    results = []
    for out, (_, expected) in zip(outputs, prompts_with_expected):
        text = out.outputs[0].text
        passed = any(e in text for e in (
            [expected] if isinstance(expected, str) else expected
        ))
        results.append((out.prompt, text, passed))
    return results


def _run_sensitivity_check(llm, params):
    """
    Check that output differs for meaningfully different inputs.
    If all outputs are identical, GDN context integration is broken.
    """
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "def hello():",
    ]
    outputs = llm.generate(prompts, params)
    token_seqs = [list(o.outputs[0].token_ids[:5]) for o in outputs]
    all_same = all(token_seqs[i] == token_seqs[0] for i in range(1, len(token_seqs)))
    return not all_same, token_seqs


@requires_model
class TestQwen35MoeGGUFTP1:
    """TP=1 coherence tests."""

    @pytest.fixture(scope="class")
    def llm(self):
        from vllm import LLM
        model = LLM(model=MODEL_PATH, **_common_kwargs(tp_size=1))
        yield model
        del model

    @pytest.fixture(scope="class")
    def params(self):
        from vllm import SamplingParams
        return SamplingParams(temperature=0.0, max_tokens=32)

    def test_tp1_france_capital(self, llm, params):
        outputs = llm.generate(["The capital of France is"], params)
        text = outputs[0].outputs[0].text
        assert "Paris" in text, f"Expected 'Paris' in {text!r}"

    def test_tp1_arithmetic(self, llm, params):
        outputs = llm.generate(["What is 2+2? Answer:"], params)
        text = outputs[0].outputs[0].text
        assert "4" in text, f"Expected '4' in {text!r}"

    def test_tp1_code_generation(self, llm, params):
        outputs = llm.generate(["def fibonacci(n):"], params)
        text = outputs[0].outputs[0].text
        assert "return" in text or "if" in text, (
            f"Expected code keywords in {text!r}"
        )

    def test_tp1_output_sensitive_to_input(self, llm, params):
        """Outputs must differ for different inputs — verifies context integration."""
        is_sensitive, token_seqs = _run_sensitivity_check(llm, params)
        assert is_sensitive, (
            f"All outputs identical — GDN not integrating context. "
            f"Token sequences: {token_seqs}"
        )


@requires_model
@requires_two_gpus
class TestQwen35MoeGGUFTP2:
    """TP=2 coherence tests (2× GPU)."""

    @pytest.fixture(scope="class")
    def llm(self):
        from vllm import LLM
        model = LLM(model=MODEL_PATH, **_common_kwargs(tp_size=2))
        yield model
        del model

    @pytest.fixture(scope="class")
    def params(self):
        from vllm import SamplingParams
        return SamplingParams(temperature=0.0, max_tokens=32)

    def test_tp2_france_capital(self, llm, params):
        outputs = llm.generate(["The capital of France is"], params)
        text = outputs[0].outputs[0].text
        assert "Paris" in text, f"Expected 'Paris' in {text!r}"

    def test_tp2_arithmetic(self, llm, params):
        outputs = llm.generate(["What is 2+2? Answer:"], params)
        text = outputs[0].outputs[0].text
        assert "4" in text, f"Expected '4' in {text!r}"

    def test_tp2_code_generation(self, llm, params):
        outputs = llm.generate(["def fibonacci(n):"], params)
        text = outputs[0].outputs[0].text
        assert "return" in text or "if" in text, (
            f"Expected code keywords in {text!r}"
        )

    def test_tp2_output_sensitive_to_input(self, llm, params):
        is_sensitive, token_seqs = _run_sensitivity_check(llm, params)
        assert is_sensitive, (
            f"All outputs identical — GDN not integrating context. "
            f"Token sequences: {token_seqs}"
        )

    def test_tp2_matches_tp1_greedy(self, params):
        """
        TP=2 greedy output must match TP=1 greedy output for the same prompt.
        (Load both models here since we need scope isolation.)
        """
        from vllm import LLM

        prompt = "The capital of France is"

        with LLM(model=MODEL_PATH, **_common_kwargs(tp_size=1)) as m1:
            out1 = m1.generate([prompt], params)[0].outputs[0].text

        with LLM(model=MODEL_PATH, **_common_kwargs(tp_size=2)) as m2:
            out2 = m2.generate([prompt], params)[0].outputs[0].text

        assert out1 == out2, (
            f"TP=1 output {out1!r} != TP=2 output {out2!r}"
        )
