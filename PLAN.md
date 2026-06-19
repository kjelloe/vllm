# Plan: Qwen3.5-35B-A3B GGUF → working vLLM inference

**Status:** 14 bugs fixed; both TP=1 and TP=2 produce incoherent garbage output.  
**Goal:** Coherent output → upstream PR merged.

---

## Phase 1 — Find the root cause (1-2 sessions)

Ordered by cost/payoff. Stop at the first hypothesis that fires.

### Step 1.1 — H9: Is `_forward_core` returning early? (5 min)

Add a single debug print to `qwen_gdn_linear_attn.py` near line 1261:

```python
def _forward_core(self, attn_metadata_raw, ...):
    import sys
    print(f"[GDN] {self.prefix} meta_is_none={attn_metadata_raw is None}", file=sys.stderr, flush=True)
    if attn_metadata_raw is None:
        ...
        return
```

Run `.venv/bin/python debugging/run.py debugging/test_tp1.py` and check stderr.

- **`meta_is_none=True` during inference** → H9 confirmed. Fix: find why `attn_metadata` is not propagated to GDN layers. (See §2.1)
- **`meta_is_none=False` during inference but output garbage** → H9 ruled out. Proceed to 1.2.
- **No `[GDN]` lines at all** → custom op dispatch failing; `no_compile_layers` lookup broken. (See §2.2)

### Step 1.2 — H13: Is `lm_head` weight uninitialized? (10 min)

Run `debugging/run.py debugging/diag_weight_audit.py` — checks:

```
lm_head is tied to embed_tokens: True/False
lm_head.weight data_ptr == embed_tokens.weight data_ptr: True/False
lm_head.weight[:3,:3] = [...]
embed_tokens norm, std: ...
```

- If `tied=False` or data_ptrs differ → Fix L or tie_word_embeddings check broken.
- If lm_head weights look random/zero → H13 confirmed.

### Step 1.3 — H12: Are `ssm_norm.weight` values correct? (5 min)

From `diag_weight_audit.py` output, check `layers[0].linear_attn.norm.weight` — should be ~1.0.

- Near-zero → norm kills all GDN output.
- Near-ones → norm OK.

### Step 1.4 — Forward hook analysis (15 min)

Run `debugging/run.py debugging/diag_forward_hooks.py` — hooks on embed_tokens, each GDN/attention layer, lm_head. Prints hidden state norms at each layer.

Expected pattern for a working model: norms stay nonzero and vary across prompts.

- Norm goes to 0 at layer N → the bug is in that layer's weight loading or forward.
- Norm is nonzero throughout but outputs are garbage → bug is in lm_head (H13) or token selection.
- Norm is constant regardless of input → GDN not integrating context (H9, H11).

### Step 1.5 — H10: blk.40 BF16 tensors missing (10 min)

Check if any MoE-related outputs are zero:

```python
# Are the full_attention layers (30-39) producing zero output?
# Hook on layers[30].forward output norm vs layers[0].forward output norm
```

Also: check HF config — does `blk.40` map to a global shared expert router?

---

## Phase 2 — Fix

Based on Phase 1 findings:

### 2.1 — If H9 confirmed (metadata not reaching GDN)

Trace `attn_metadata` flow from vLLM engine → `Qwen3_5MoeForConditionalGeneration.forward` → `Qwen3_5Model.forward` → each `Qwen3_5DecoderLayer.forward`. The metadata must reach `QwenGatedDeltaNetAttention.forward_cuda`, which passes it via the custom op to `_forward_core`.

Check `attn_metadata` type: vLLM v1 uses `FlashAttentionMetadata` for full-attention layers. GDN (`linear_attention`) layers use a different metadata type (`QwenGDNMetadata` or similar). If the dispatcher passes `None` to GDN layers because it doesn't recognize the layer type, that's the bug.

Look at how the engine calls forward: does it pass per-layer metadata or a single unified object?

### 2.2 — If lm_head untied (H13)

Check `gguf_loader.py` line ~520:
```python
if any("lm_head.weight" in n for n in all_extra_names):
    model_config.hf_config.update({"tie_word_embeddings": True})
```

`all_extra_names` contains the HF-namespace names of tensors NOT present in the GGUF (i.e., weights that should be tied). After Fix F, the name for lm_head in `gguf_to_hf_name_map` is `model.language_model.lm_head.weight`. Does the substring `"lm_head.weight"` appear in this? Yes — `"lm_head.weight"` is a substring of `"model.language_model.lm_head.weight"`. So this should work.

If H13 fires anyway, look at `all_extra_names` to see what names are in it at runtime.

### 2.3 — If ssm_norm wrong (H12)

Check the GGUF name map entry for `blk.N.ssm_norm.weight`. After Fix F, it should become `model.language_model.layers.N.linear_attn.norm.weight`. Verify that this vLLM module path is correct for the GDN `norm` attribute (which uses `RMSNormGated`).

---

## Phase 3 — Validate

1. Remove any debug prints added in Phase 1.
2. Run `.venv/bin/python debugging/run.py debugging/test_tp1.py` → must pass all 3 prompts.
3. Run `.venv/bin/python debugging/run.py debugging/test_sensitivity.py` → outputs must differ for different inputs.
4. Run `.venv/bin/python debugging/run.py debugging/test_qwen35moe_tp2.py` → must pass all 3 prompts.
5. Run pre-commit linters on changed files.

---

## Phase 4 — PR

1. Update `pr.md` test section with passing results.
2. Update `BUG_HYPOTHESIS.md` marking root cause as CONFIRMED + FIXED.
3. `gh pr create --repo vllm-project/vllm ...` using `pr.md` as body.
4. Respond to reviewer feedback; re-run tests if requested.

---

## New hypotheses (H13-H15)

See `BUG_HYPOTHESIS.md` for full detail.

| # | Summary | Diagnosis tool |
|---|---------|---------------|
| H9 | `_forward_core` returns early (attn_metadata is None) | Debug print in `_forward_core` |
| H12 | `ssm_norm.weight` not loaded → zero scale on GDN output | `debugging/diag_weight_audit.py` |
| H13 | `lm_head` not tied → uninitialized output logit weights | `debugging/diag_weight_audit.py` |
| H14 | `in_proj_ba` F32 gate values wrong/saturated → GDN gate kills output | `debugging/diag_weight_audit.py` |
| H15 | attn_metadata type mismatch for GDN vs full_attention dispatch | Debug print in `_forward_core` |

---

## GPU device selection

| Test type | `CUDA_VISIBLE_DEVICES` | `gpu_memory_utilization` |
|-----------|------------------------|--------------------------|
| TP=1 | `0` (RTX 4090 only) | 0.935 |
| TP=2 | `0,1` (4090 + 3090) | 0.85 |

Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside `CUDA_VISIBLE_DEVICES`. Use `setdefault` so the user can override from the shell. Without explicit device selection, vLLM sees both GPUs even for TP=1 and miscalculates available VRAM.

---

## Diagnostic scripts

| Script | Purpose | Requires GPU |
|--------|---------|-------------|
| `debugging/run.py <script>` | Subprocess wrapper (vLLM v1 `__main__` guard) | No |
| `debugging/test_tp1.py` | TP=1 coherence (3 prompts) | Yes |
| `debugging/test_qwen35moe_tp2.py` | TP=2 coherence (3 prompts) | Yes (2×) |
| `debugging/test_sensitivity.py` | Check output varies with input | Yes |
| `debugging/diag_weight_audit.py` | Runtime: quant_method, ssm_norm, lm_head tie | Yes |
| `debugging/diag_forward_hooks.py` | Runtime: hidden state norms at each layer | Yes |
| `debugging/check_conv1d.py` | Raw GGUF conv1d shape (no GPU) | No |
| `debugging/check_embed_direct.py` | Raw GGUF embed + F32 values (no GPU) | No |

## Test suite (./tests)

| Test file | What it covers | Needs GPU |
|-----------|----------------|----------|
| `tests/models/quantization/test_gguf_qwen35moe.py` | Name mapping, dt_bias, prefix, tie logic | No |
| `tests/kernels/mamba/test_gdn_weight_loading.py` | `mamba_v2_sharded_weight_loader` conv1d | Yes (small) |
| `tests/models/quantization/test_gguf.py` | Existing: add Qwen3.5-MoE config | Yes (needs model) |
