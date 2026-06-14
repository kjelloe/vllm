"""
Coherence test for Qwen3.5-35B-A3B Q4_K_M GGUF.

Usage:
    python3 ./test_qwen35moe_tp2.py          # TP=2 (default)
    python3 ./test_qwen35moe_tp2.py --tp1    # TP=1 sanity baseline (single GPU)
"""
import os
import sys

# Ensure stable GPU ordering across CUDA and vLLM (required for mixed-arch TP).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

tp_size = 1 if "--tp1" in sys.argv else 2

model_path = os.path.join(
    os.environ.get("LLAMA_MODELS_DIR", ""),
    "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf",
)
if not os.path.exists(model_path):
    sys.exit(f"ERROR: model not found at {model_path}\n"
             "Set $LLAMA_MODELS_DIR correctly.")

print(f"Loading: {model_path}")

from vllm import LLM, SamplingParams  # noqa: E402

if __name__ == "__main__":
    print(f"Running with tensor_parallel_size={tp_size}")
    llm = LLM(
        model=model_path,
        tokenizer="Qwen/Qwen3.5-35B-A3B",
        hf_config_path="Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=tp_size,
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
    label = f"TP={tp_size} COHERENCE TEST"
    if passed:
        print(f"=== {label}: PASSED ===")
        if tp_size == 2:
            print("Ready to open upstream PR.")
    else:
        print(f"=== {label}: FAILED ===")
        print("Check output above for weight-skip warnings or NaN in logits.")
        sys.exit(1)
