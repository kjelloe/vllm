# Bug Hypothesis: Garbage output from Qwen3.5-35B-A3B Q4_K_M GGUF

**Date:** 2026-06-18  
**Status:** Under investigation — root cause not yet found  
**Symptoms:** Model loads cleanly, inference runs, all outputs are incoherent garbage on both TP=1 and TP=2.  
Examples: `"The capital of France is"` → `" ELECT mistakenchol TELE raise为王..."`, `"2+2"` → `"elizfindByrimp..."`

---

## What has been confirmed OK

These were traced through the code or tested directly and are NOT the root cause:

- **Name mapping** — GGUF names correctly map to HF names via `gguf_to_hf_name_map` (traced through `_get_gguf_weights_map`)
- **Prefix routing** — `hf_to_vllm_mapper` + `AutoWeightsLoader` descent correctly strips prefixes before calling `Qwen3_5Model.load_weights`
- **stacked_params_mapping** — `in_proj_qkv/z → in_proj_qkvz` and `in_proj_b/a → in_proj_ba` find correct parameters in `params_dict`
- **GGUF data shape + TP split** — `output_dim=0` on packed data tensor is correct; TP=2 splits produce correct per-rank sizes
- **A_log, dt_bias** — F32 parameters load via direct path; `sharded_weight_loader` handles them correctly
- **conv1d weight** — F32 full 8192-row weight confirmed (`blk.0.ssm_conv1d.weight` = 8192 × 4 = 32768 F32 elements = `conv_dim × kernel_size`)
- **conv1d loading** — `mamba_v2_sharded_weight_loader` with 2D→3D unsqueeze loads correctly
- **in_proj_ba** — `UnquantizedLinearMethod` (Fix 2 in `is_layer_skipped_gguf`) creates `weight` not `qweight`; F32 loads correctly
- **MoE expert weights** — 3D batched tensors load via `FusedMoE.weight_loader` full_load path
- **embed_tokens dequantization** — `ggml_dequantize` returns sane values for Q4_K (tested in isolation: `std > 0.001`, `|max| < 10`; token 760 "The" returns valid BF16 embedding)
- **F32 weight values** — `ssm_a: [-72.33, -0.0186]`, `ssm_dt.bias` and `output_norm.weight` in expected ranges
- **static_forward_context registration** — GDN layers register via `compilation_config.static_forward_context[prefix] = self` correctly traced through `qwen_gdn_linear_attn.py:561-564`
- **is_layer_skipped_gguf chain** — `apply_vllm_mapper` is called before `__init__`; F32 layers appear to correctly get `UnquantizedLinearMethod`
- **TP=1 also fails** (H1 CONFIRMED) — garbage output occurs on TP=1; bug is not TP-specific

---

## What has been ruled out

- **H4 — GDN Triton kernel SM89/SM86 specific** — RULED OUT: TP=1 also fails
- **H5 — conv1d weight shape wrong for TP=2** — RULED OUT: TP=1 also fails; and conv1d weight shape confirmed correct
- **H6 — SSM recurrent state wrong shape for TP=2** — RULED OUT: TP=1 also fails
- **H7 — conv1d zero-V (only Q+K stored)** — RULED OUT: full 8192-element conv1d confirmed

---

## Active hypotheses

### H9 — `attn_metadata_raw is None` during inference → GDN returns early (HIGH PRIORITY)

**Root cause:** `_forward_core` (line 1261) starts with:
```python
if attn_metadata_raw is None:
    self._warmup_prefill_kernels(...)
    return   # ← returns without writing to core_attn_out
```
`core_attn_out` is initialized to zeros before the `torch.ops.vllm.qwen_gdn_attention_core` call (line ~940). If the custom op's `_forward_core` returns early (e.g. because `attn_metadata_raw` is None during actual inference due to a metadata propagation bug), all 30 GDN layers silently produce zero attention output. The model then falls back to pure residual + z-gate output, which is incoherent garbage.

**Why likely:** No crash, no NaN, but completely incoherent output — consistent with all GDN layers silently no-oping.  
**Note:** `compilation_config.static_forward_context` lookup could also fail silently if `layer_name` lookup in `no_compile_layers` raises or returns wrong layer.  
**Diagnosis:**
```python
# Add temporary print to qwen_gdn_linear_attn.py _forward_core:
print(f"[GDN _forward_core] layer={self.prefix}, attn_meta={attn_metadata_raw is not None}")
```

---

### H10 — `blk.40` BF16 tensors at layer index 40 (outside 0-39 range)

**Root cause:** GGUF contains `blk.40.ffn_gate_inp.weight` and `blk.40.ffn_gate_inp_shexp.weight` as BF16 tensors at index 40. The model has 40 layers (0-39), so these fall outside the `range(num_hidden_layers)` iteration in `gguf_to_hf_name_map`. They are never mapped and never loaded.

**Impact assessment:** These are MoE gate inputs for what may be a "layer 40" shared expert routing. If the model actually needs these, the MoE router for a layer is receiving zero/garbage weights → that layer produces wrong output → propagates through all subsequent layers.  
**Evidence against:** The 10 full_attention layers (30-39) should work even without this — the gate weight affects only one router. But if this IS the shared/global expert router weight, impact is high.  
**Diagnosis:**
```python
# Check if GGUF layer index 40 corresponds to some HF named tensor
# or if it should map to blk.39 or a global weight
reader = GGUFReader(model_path)
for t in reader.tensors:
    if 'blk.40' in t.name:
        print(f"{t.name}: {t.tensor_type.name} shape={t.shape}")
```

---

### H11 — GDN SSM/conv state slot indices wrong → stale or zero recurrent state

**Root cause:** GDN layers maintain a `conv_state` and `ssm_state` per sequence. These states are indexed by layer index. If the state slot assignment is off (e.g., layer uses wrong index into the KV cache state pool, or state shape mismatch), the GDN reads stale/zero state for every step, making the recurrent accumulation fail.

**Evidence for:** The GDN custom op (`qwen_gdn_attention_core`) reads state from the `forward_context.no_compile_layers[layer_name]._forward_core` path. If `layer_name` lookup fails or returns a different layer's state, state cross-contamination or zeros follow.  
**Diagnosis:**
```python
# Check get_mamba_state_shape_from_config return values match what _forward_core expects
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
# Introspect layers[0].linear_attn._forward_core state shapes
```

---

### H12 — `norm` (RMSNormGated) weight loading failed → output scaled to zero

**Root cause:** `_output_projection` in GDN forward passes `core_attn_out` through `self.norm` (an `RMSNormGated` layer) gated with `z`. If the `norm.weight` failed to load (wrong name mapping for `blk.N.ssm_norm.weight`), the output is scaled by uninitialized/zero weights → zero or garbage after all 30 GDN layers.

**Evidence for:** `blk.0.ssm_norm.weight` is a small F32 tensor that uses a non-standard GGUF name; if the name isn't in the map or maps to a wrong HF path, it silently loads with `weight = ones` (RMSNorm default init) or zeros.  
**Evidence against:** `output_norm.weight` was confirmed sane; but `ssm_norm.weight` (per-layer) wasn't checked.  
**Diagnosis:**
```python
# In check_weight_stats: print ssm_norm.weight for layer 0 after loading
print(m.layers[0].linear_attn.norm.weight[:8])  # should be ~1.0 if GGUF F32 loaded
```

---

### H13 — `lm_head` weight not tied → uninitialized output logit matrix (HIGH)

**Root cause:** The GGUF does not contain `output.weight` (= `lm_head.weight`) because `tie_word_embeddings=True` in the HF config. `gguf_loader.py` detects this via:
```python
if any("lm_head.weight" in n for n in all_extra_names):
    model_config.hf_config.update({"tie_word_embeddings": True})
```
After Fix F, the HF-namespace name for lm_head in `gguf_to_hf_name_map` is `model.language_model.lm_head.weight`. The substring `"lm_head.weight"` IS present in this string, so the check should work. But if `all_extra_names` is empty or doesn't include the lm_head entry for any reason, `tie_word_embeddings` stays at its default (False for qwen3_5_moe?), and `lm_head.weight` is never initialized from `embed_tokens`. Every next-token logit distribution is then uniform/random → garbage output on ALL prompts.

**This is architecturally critical:** garbage logits = garbage output regardless of how good the hidden states are.  
**Diagnosis:**
```python
m = llm.llm_engine.driver_worker.model_runner.model
lm = m.language_model
print("lm_head same as embed:", lm.lm_head.weight.data_ptr() == lm.model.embed_tokens.weight.data_ptr())
print("lm_head weight norm:", lm.lm_head.weight.float().norm().item())
```
If `lm_head` norm is very small or the pointer differs → H13 confirmed.

---

### H14 — `in_proj_ba` (b/a gate values) wrong → GDN recurrence saturated

**Root cause:** `in_proj_ba` is an F32 `MergedColumnParallelLinear` containing the input-dependent recurrent coefficients (b = input mixing, a = forget gate). These load via `UnquantizedLinearMethod`. If the GGUF name for `ssm_alpha.weight` (→ `in_proj_a`) or `ssm_beta.weight` (→ `in_proj_b`) is misrouted to the wrong shard or wrong parameter:
- `a` values near zero → GDN forgets immediately → output is zero every step
- `a` values near one → GDN never forgets → state saturates → garbage after a few tokens

**Evidence for:** b/a values come from F32 tensors loaded via a separate path; any shard index error in `stacked_params_mapping` for `in_proj_ba` would not crash but silently produce wrong values.  
**Diagnosis:**
```python
ba_weight = m.layers[0].linear_attn.in_proj_ba.weight
print("in_proj_ba shape:", ba_weight.shape)  # should be [32, 2048] for TP=1
b_part = ba_weight[:16, :]  # b values
a_part = ba_weight[16:, :]  # a values
print("b range:", b_part.min().item(), b_part.max().item())
print("a range:", a_part.min().item(), a_part.max().item())
```

---

### H15 — attn_metadata wrong type for GDN → dispatch returns wrong object

**Root cause:** vLLM v1 may dispatch different `attn_metadata` objects for different layer types. If `Qwen3_5DecoderLayer.forward` passes the full-attention metadata object to GDN layers (or vice versa), `_forward_core` receives a metadata with wrong fields — e.g., `seq_lens` missing or `block_tables` None. The code might check for specific attributes to decide whether to run the GDN, and a wrong-type metadata could trigger the early-return path at line 1261.

**Relationship to H9:** H15 is a specific mechanism for H9. If `attn_metadata_raw is None` fires, H15 explains WHY it's None even though metadata exists — the dispatcher may be passing `None` specifically to linear_attention layers.

**Diagnosis:** Same debug print as H9, but also print `type(attn_metadata_raw)` to see what's being passed.

---

### H2 — embed_tokens quant_method not verified at runtime (LOW PRIORITY)

**Root cause:** Fix L passes `quant_config` to `VocabParallelEmbedding`. Code trace shows Fix L is present and `ggml_dequantize` works in isolation. But we haven't runtime-confirmed that `type(m.embed_tokens.quant_method).__name__ == 'GGUFEmbeddingMethod'`. If Fix L's code path has a subtle bug (e.g., `quant_config` is None at construction time), embeddings would be wrong.

**Current assessment:** Likely NOT the root cause — code path traced and appears correct, and embed_tokens dequantize works in raw test.  
**Diagnosis:**
```python
from vllm import LLM
llm = LLM(model=..., enforce_eager=True)
m = llm.llm_engine.driver_worker.model_runner.model
print(type(m.language_model.model.embed_tokens.quant_method).__name__)
```

---

### H3 — `is_layer_skipped_gguf` namespace mismatch (CONFIRMED BUG, impact unclear)

**Status: CONFIRMED** — `is_layer_skipped_gguf` returns `False` for all Qwen3.5-MoE F32 layers due to namespace mismatch. Whether this causes garbage output depends on whether `GGUFLinearMethod.apply()` correctly handles F32 data via the UNQUANTIZED_TYPES fallback.

**Root cause:** After Fix F, `unquantized_modules` entries carry `model.language_model.layers.0...` prefix (HF namespace). `is_layer_skipped_gguf` is called with vLLM module prefix `language_model.model.layers.0...`. The check `module_name in shard_prefix` is False in both directions — neither is a substring of the other.

**Consequence:** F32 layers in GDN (in_proj_ba, conv1d, ssm_norm) get `GGUFLinearMethod` instead of `UnquantizedLinearMethod`. This was confirmed via direct call:
```python
is_layer_skipped_gguf("language_model.model.layers.0.linear_attn.in_proj_ba",
                      ["model.language_model.layers.0.linear_attn.in_proj_ba"])
# Returns False
```

**Key question:** `GGUFLinearMethod.apply()` passes qweight through `ggml_dequantize(F32)` which is a near-no-op. If the F32 weight data is correctly stored in `qweight`, the forward pass would still be correct. The bug may be in weight LOADING rather than forward.

**Pinned in test:** `tests/models/quantization/test_gguf_qwen35moe.py::TestIsLayerSkippedGGUF::test_h3_namespace_mismatch_for_qwen35moe_f32_layers`

**Fix:** Change Fix F to replace `model.` → `` (strip it) rather than adding a prefix, OR fix `is_layer_skipped_gguf` to check both directions.

---

## Recommended investigation order

1. **H9 (HIGHEST PRIORITY): Add print to `_forward_core`** — check if `attn_metadata_raw` is None during actual inference. One line of debug output settles this.
2. **H12: Check `ssm_norm.weight` values at runtime** — if near-zeros instead of ~1.0, norm weight loading failed.
3. **H10: Check blk.40 tensor mapping** — are these tensors mapped anywhere? Missing them = missing MoE gate for certain layers.
4. **H11: Check state slot indices** — harder to probe; come back if H9/H12 don't explain it.
5. **H2 / H3: Runtime quant_method check** — if all above OK, verify embed_tokens and per-layer quant methods at runtime.

---

## Diagnostic scripts

```bash
# TP=1 coherence test
.venv/bin/python /tmp/run.py /tmp/test_tp1.py

# Sensitivity analysis (same output for all inputs = context ignored)
.venv/bin/python /tmp/run.py /tmp/test_sensitivity.py

# Raw GGUF + dequantize check (no vLLM model)
.venv/bin/python /tmp/check_embed_direct.py

# Conv1d weight shape check (raw GGUF)
.venv/bin/python /tmp/check_conv1d.py
```

### H9 debug patch (temporary)

In `/home/kjelloe/GIT/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` near line 1261:

```python
def _forward_core(self, attn_metadata_raw, ...):
    import sys
    print(f"[GDN] layer={self.prefix} meta_is_none={attn_metadata_raw is None}", file=sys.stderr, flush=True)
    if attn_metadata_raw is None:
        self._warmup_prefill_kernels(...)
        return
    # ... rest of forward
```

Then run `/tmp/test_tp1.py` and look for `[GDN]` lines. If `meta_is_none=True` during inference, H9 is confirmed.
