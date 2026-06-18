# Bug Hypothesis: Garbage output from Qwen3.5-35B-A3B Q4_K_M GGUF

**Date:** 2026-06-18  
**Status:** Under investigation  
**Symptoms:** Model loads cleanly, inference runs, but all outputs are incoherent garbage.  
Examples: `"The capital of France is"` → `" ELECT mistakenchol TELE raise为王..."`, `"2+2"` → `"elizfindByrimp..."`

---

## What has been ruled out

All of the following have been traced through the code and verified correct:

- **Name mapping** — GGUF names correctly map to HF names via `gguf_to_hf_name_map` (tested by running `_get_gguf_weights_map` diagnostically)
- **Prefix routing** — `hf_to_vllm_mapper` + `AutoWeightsLoader` descent correctly strips prefixes before calling `Qwen3_5Model.load_weights`
- **stacked_params_mapping** — `in_proj_qkv/z → in_proj_qkvz` and `in_proj_b/a → in_proj_ba` replacements find the correct parameters in `params_dict`
- **GGUF data shape + TP split** — `output_dim=0` on the packed data tensor is correct (data shape `[n_output_rows, packed_bytes]`); TP=2 splits produce correct per-rank sizes
- **A_log, dt_bias** — F32 parameters load via direct path; `sharded_weight_loader` handles them
- **conv1d** — F32 weight loads via `mamba_v2_sharded_weight_loader` with 2D→3D unsqueeze
- **in_proj_ba** — `UnquantizedLinearMethod` (after Fix 2 in `is_layer_skipped_gguf`) creates `weight` not `qweight`; F32 loads correctly via `MergedColumnParallelLinear.weight_loader`
- **MoE expert weights** — 3D batched tensors load via `FusedMoE.weight_loader` full_load path

---

## Hypotheses (ranked by likelihood)

### H1 — TP=1 also produces garbage (most likely if not yet tested)

**Prediction:** Running `--tp1` produces identical garbage.  
**Implication:** The bug is NOT in TP communication or sharding — it's in weight loading or the model forward pass.  
**Diagnosis:** Run `/tmp/run.py /tmp/test_tp1.py`

---

### H2 — VocabParallelEmbedding `quant_config` not reaching `embed_tokens`

**Root cause:** Fix L passed `quant_config` to `VocabParallelEmbedding`. If this is missing or ineffective in the current branch, `embed_tokens` uses `UnquantizedEmbeddingMethod` → the Q4_K token embedding is **silently skipped** (weight_loader finds `weight` param, but GGUF yields `qweight` → no match → warning_once and `continue`). Input embeddings are then zero-initialized or uninitialized.

**Evidence for:** The output is incoherent across ALL prompts equally — consistent with garbage embeddings.  
**Evidence against:** Fix L is supposed to be in the branch.  
**Diagnosis:**
```python
# in check_weight_stats.py
print(type(m.embed_tokens.quant_method).__name__)  # should be GGUFEmbeddingMethod
print(getattr(m.embed_tokens, 'qweight', 'MISSING'))  # should be a tensor
print(getattr(m.embed_tokens, 'weight', 'MISSING'))   # should be MISSING
```

---

### H3 — `is_layer_skipped_gguf` still wrong for some layers (residual after Fix 2)

**Root cause:** Fix 2 changed the direction of the `is_layer_skipped` check. But the `unquantized_modules` list in `GGUFConfig` is built from the HF-namespace names in `weight_type_map`, which (after Fix F) carry a `model.language_model.` prefix. Meanwhile, `is_layer_skipped_gguf` is called with the vLLM module prefix (e.g., `language_model.model.layers.0.linear_attn.in_proj_ba`). The two namespaces may still not match cleanly for all layers.

**Implication:** Some F32 linear layers (like `norm` or `out_proj`) may get `GGUFLinearMethod` instead of `UnquantizedLinearMethod`, causing weight type mismatch — F32 data stored in a GGUF-quantized parameter or vice versa.

**Diagnosis:** Run `/tmp/check_layer3.py` and compare expected vs actual quant methods per layer.

---

### H4 — GDN Triton/FLA kernel produces wrong output on SM89/SM86

**Root cause:** The `fla_chunk_gated_delta_rule` Triton kernel (used for prefill on non-Hopper GPUs) has a latent numerical bug specific to SM89 (RTX 4090) or SM86 (RTX 3090) that causes silent wrong output rather than a crash. The 30 GDN layers all process hidden states through this path; a bug there would corrupt all outputs.

**Evidence for:** No crash, no NaN, but coherent garbage. The GDN kernel is a relatively new, complex Triton implementation not widely tested on these exact GPU pairs.  
**Evidence against:** Triton kernels are generally architecture-agnostic.  
**Diagnosis:** Test with TP=1 on just the 3090 (SM86), then just the 4090 (SM89). If one passes and one fails, it's hardware-specific. If both fail, rule out hardware. If TP=1 passes but TP=2 fails, rule this out.

---

### H5 — conv1d weight shape wrong for TP=2

**Root cause:** `mamba_v2_sharded_weight_loader` is built with a `shard_spec` that assumes specific groupings of the Q/K/V heads. If the shard_spec for the GDN conv1d doesn't match the actual Q+K+V layout after TP splitting, the conv1d processes the wrong channels.

**Evidence for:** The conv1d state is central to the GDN recurrent computation; wrong channels = wrong hidden state = garbage.  
**Diagnosis:** After loading, check `m.layers[0].linear_attn.conv1d.weight.shape`. Should be `[4096, 1, 4]` for TP=2 (4096 channels per rank).

---

### H6 — SSM recurrent state not initialized / wrong shape for TP=2

**Root cause:** The mamba/GDN SSM state (conv1d state buffer) is allocated at startup based on TP-aware shapes. If the allocation uses `num_v_heads` (total=32) instead of `num_v_heads // tp_size` (16 per rank), the recurrent state has wrong dimensions and the kernel overwrites adjacent memory or silently produces zeros.

**Diagnosis:** Check `get_mamba_state_shape_from_config` return values and verify the allocated state shape matches what the GDN forward pass expects.

---

## Recommended investigation order

1. **Run TP=1** → eliminates H4, H5, H6 if it passes; confirms H1-H3 if it also fails
2. **Check `embed_tokens.quant_method`** → verify Fix L is active (H2)
3. **Run `check_weight_stats.py --tp1`** → check NaN/zero fractions and quant method types (H3)
4. If TP=1 fails: **Add debug print in `Qwen3_5Model.load_weights`** to log each weight name that hits the `name not in params_dict` path → shows any silently skipped weights (H2, H3)
5. If TP=1 passes: **Compare TP=2 vs TP=1 outputs for a single GDN layer** by intercepting `linear_attn.forward()` output on rank 0

---

## Diagnostic scripts

```bash
# TP=1 coherence test
.venv/bin/python /tmp/run.py /tmp/test_tp1.py

# Weight stats (quant types, nan/zero, shapes)
.venv/bin/python /tmp/run.py /tmp/check_weight_stats.py --tp1

# Weight stats TP=2
.venv/bin/python /tmp/run.py /tmp/check_weight_stats.py
```
