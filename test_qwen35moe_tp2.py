"""
Coherence test for Qwen3.5-35B-A3B Q4_K_M GGUF with tensor_parallel_size=2.
Run after enabling both RTX 3090s in WSL2.

Usage:
    export PATH="/mnt/c/GIT/vllm/.venv/bin:$PATH"
    python3 /tmp/test_qwen35moe_tp2.py
"""
import os
import sys

model_path = os.path.join(
    os.environ.get("LLAMA_MODELS_DIR", ""),
    "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf",
)
if not os.path.exists(model_path):
    sys.exit(f"ERROR: model not found at {model_path}\n"
             "Set $LLAMA_MODELS_DIR correctly.")

print(f"Loading: {model_path}")

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(
    model=model_path,
    tokenizer="Qwen/Qwen3.5-35B-A3B",
    hf_config_path="Qwen/Qwen3.5-35B-A3B",
    tensor_parallel_size=2,
    enforce_eager=True,
    max_model_len=512,
    gpu_memory_utilization=0.90,
    disable_log_stats=True,
)

params = SamplingParams(temperature=0.0, max_tokens=64)

prompts = [
    ("2+2", "What is 2+2? Answer with just the number."),
    ("france", "The capital of France is"),
    ("python", "def fibonacci(n):"),
]

outputs = llm.generate([p for _, p in prompts], params)

print("\n=== Results ===")
results = {}
for (key, prompt), out in zip(prompts, outputs):
    text = out.outputs[0].text
    print(f"  Prompt: {prompt!r}")
    print(f"  Output: {text!r}\n")
    results[key] = text

# Coherence checks
passed = True

if "4" not in results["2+2"]:
    print("FAIL: 2+2 answer does not contain '4'")
    passed = False
else:
    print("PASS: 2+2 answer contains '4'")

if "Paris" not in results["france"]:
    print("FAIL: France capital does not contain 'Paris'")
    passed = False
else:
    print("PASS: France capital contains 'Paris'")

if "return" not in results["python"] and "if" not in results["python"]:
    print("FAIL: fibonacci output looks incoherent")
    passed = False
else:
    print("PASS: fibonacci output looks coherent")

print()
if passed:
    print("=== TP=2 COHERENCE TEST: PASSED ===")
    print("Ready to open upstream PR.")
else:
    print("=== TP=2 COHERENCE TEST: FAILED ===")
    print("Check output above for weight-skip warnings or NaN in logits.")
    sys.exit(1)
