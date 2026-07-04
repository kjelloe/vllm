# Bug Hypothesis: Garbage output from Qwen3.5-35B-A3B Q4_K_M GGUF

**Date:** 2026-07-04 (updated)  
**Status:** ROOT CAUSE FOUND (2026-07-04) — GGUF stores GDN V-heads in ggml *tiled* order; vLLM expects HF *grouped* order and never converts. See "ROOT CAUSE" section below.  
**NOTE (2026-07-03):** Branch merged with upstream/main. `Qwen3_5Model.load_weights` no longer uses a manual `stacked_params_mapping` loop; it now goes through `AutoWeightsLoader` with `hf_to_vllm_mapper` (`orig_to_new_stacked` at `vllm/model_executor/models/qwen3_5.py:203-210`). Same shard ids as before (`in_proj_qkv`→`(0,1,2)`, `in_proj_z`→`3`, `in_proj_b`→`0`, `in_proj_a`→`1`), but the code path is different — see Plan 0.  
**Symptoms:** Model loads cleanly, inference runs, all outputs are incoherent garbage on both TP=1 and TP=2.  
Examples: `"The capital of France is"` → `" is"` (top token), `" capital"` (rank 2), `" Cap"` (rank 3) — context-word copying pattern.  
Paris rank ~1303 vs expected rank ~1.

---

## SESSION STATUS & NEXT ACTIONS (2026-07-04)

**Root cause: FOUND and triple-confirmed.** GDN V-head tiling mismatch (see ROOT
CAUSE section). Confirmed by (1) independent source analysis, (2) closed plugin
PR #31's identical implementation, (3) empirical packed-layout check.

**Fix home:** the out-of-tree `vllm-gguf-plugin` (GGUF is no longer in vLLM core —
see STRUCTURAL CHANGE section). Fix = a self-contained `Qwen3_5GGUFAdapter`.

**Next actions, in order:**
1. [DONE 2026-07-04] Removed orphaned in-tree GGUF dead files
   (`gguf_loader.py`, `quantization/gguf.py`) — staged, not committed. vLLM
   still imports cleanly.
2. [DONE 2026-07-04] Wrote `Qwen3_5GGUFAdapter` in
   `vllm_gguf_plugin/weights_adapter/qwen3_5.py` (clone + site-packages 0.0.2),
   registered in `_ADAPTER_REGISTRY`. Combines #31's verified `transform_weight`
   (V-head reorder) with my proven `build_name_map` (arch remap
   `qwen3_5_moe`→`qwen35moe`, text-only forcing, MoE + ssm_dt sideload,
   `model.language_model.` prefix). **Offline name-map dry-run PASSED**: all HF
   params map; 733/753 GGUF tensors covered; only 20 uncovered are `blk.40.*`
   (MTP head, benign). GDN mappings all correct.
3. [IN PROGRESS] Validate on Qwen3.5-35B-A3B Q4_K_M (TP=1, fp16, max_len=256,
   gpu_util=0.90). Iterating through load-path issues:
   - **Prefix bug (FIXED):** the vLLM multimodal wrapper nests the whole text
     model under `language_model.` (`language_model.model.layers.*`,
     `language_model.lm_head.*`), NOT `model.language_model.*` as old Fix F
     assumed. `build_name_map` now prepends `language_model.`.
   - **Wrapper detection (FIXED):** `uses_cg_wrapper` was keyed on
     `config.architectures`, which the GGUF config path does not populate →
     Fix F was skipped at load time. Re-keyed on `vision_config` presence (the
     real multimodal-wrapper signal). Offline cross-check vs the 1258 real
     vLLM params: 0 mismatches across all 733 real-tensor mappings.
   - **Tuple shard_id (IN PROGRESS):** Qwen3.5's `in_proj_qkvz` loads the
     combined GGUF `attn_qkv` via a tuple shard_id `(0,1,2)` (baked into
     vLLM's `Qwen3_5Model` hf_to_vllm_mapper `orig_to_new_stacked`; this was
     old in-tree Fix G). The plugin's `quantization/params.py`
     `_gguf_weight_type_loader_v2` only special-cases `shard_id is None`; for a
     tuple it delegates to vLLM's `_load_fused_module_from_checkpoint`, which
     assumes a real weight with `output_dim` → `AttributeError` on the GGUF
     `GGUFUninitializedWeightTypeParameter`. `_store_gguf_weight_type` /
     `_gguf_shard_id_as_int` also can't handle a tuple. **Fix:** teach the
     plugin's GGUF weight-type (and weight) loaders to handle tuple shard_ids
     by storing/placing the value across each listed shard. This expands the
     change into plugin core (`quantization/params.py`), re-homing Fix G.
   - **Tuple shard_id (FIXED):** patched plugin
     `quantization/params.py:_gguf_weight_type_loader_v2` to store the
     weight-type per shard for a tuple shard_id (real `.qweight` already loads
     fine — `GGUFWeightParameter` has `output_dim` and row-narrows cleanly).
   - **RoutedExperts MoE (IN PROGRESS):** next error is
     `IndexError: index 0 is out of bounds for dimension 0 with size 0` at
     `fused_moe/routed_experts.py:692` (`param.data[expert_id]` on empty data).
     GGUF expert tensors are 3D packed `(256 experts, 512, 1152)` so
     `full_load` should be True, but the `RoutedExperts` param isn't
     materialized. Looks like a gap between the plugin's GGUF MoE
     materialization (`_materialize_gguf_moe_param`) and vLLM's newer
     `RoutedExperts` abstraction that Qwen3.5 uses. Under investigation.

   **Scope note:** supporting this model in the plugin has required 3+
   plugin-core fixes (adapter + params.py tuple-shard + MoE), not just the
   adapter. Count of remaining issues unknown. My old in-tree Fixes A–M
   handled all of this end-to-end (model loaded and ran); the plugin is the
   right long-term home but is catching up on tuple-shard GDN + RoutedExperts.

   This is the real test of the V-head fix (#31's GDN path was never exercised
   — its smoke test was GDN-less Qwen3-0.6B). Runner: `/tmp/run.sh <script.py>`
   (fixed env; matches the `Bash(/tmp/*)` allow rule).
4. PR to vllm-gguf-plugin ("add Qwen3.5 support"), attributing prior art #31.
   **Gated on explicit user approval — do not push/open without it.**

**Env note:** plugin editable install fails (compiled `_C_gguf`); edit the
pure-Python adapter in `.venv/.../vllm_gguf_plugin/` for testing.

---

## What has been confirmed OK

- **Name mapping** — all 40 model layers mapped; `blk.40` (speculative NEFT head) unmapped but irrelevant. Verified via `gguf.get_tensor_name_map(QWEN35MOE, 40)`.
- **Gate/router weights** — `blk.N.ffn_gate_inp.weight` → `model.layers.N.mlp.gate.weight` via gguf-py Priority-3 path.
- **Shared expert weights** — `ffn_{gate,up,down}_shexp.weight` all mapped for QWEN35MOE.
- **Prefix routing** — `hf_to_vllm_mapper` + `AutoWeightsLoader` descent correctly strips prefixes before calling `Qwen3_5Model.load_weights`. (`vllm/model_executor/model_loader/gguf_loader.py:409-421`)
- **Stacked shard mapping** — `in_proj_qkv`→`in_proj_qkvz` shard `(0,1,2)`, `in_proj_z`→shard `3`, `in_proj_b`→`in_proj_ba` shard `0`, `in_proj_a`→shard `1`. Verified under the old manual loop; post-merge this lives in `hf_to_vllm_mapper.orig_to_new_stacked` (`vllm/model_executor/models/qwen3_5.py:203-210`) — needs re-verification (Plan 0).
- **Full-attention qkv_proj weight values** — Layer 3 `qkv_proj` cosine sim 0.999984 vs dequantize reference; weights correctly loaded via `q_proj`→shard "q", `k_proj`→"k", `v_proj`→"v".
- **Full-attention HF state_dict shapes** — `q_proj [8192,2048]` (32 heads = 16 Q + 16 gate), `k_proj [512,2048]`, `v_proj [512,2048]`, `o_proj [2048,4096]`.
- **Full-attention GGUF tensors (blk.3)** — `attn_q [2048,8192]`, `attn_k [2048,512]`, `attn_v [2048,512]`, `attn_output [4096,2048]`. Shapes consistent with HF.
- **in_proj_qkvz shard ordering** — `output_sizes=[key_dim, key_dim, value_dim, value_dim]` = [Q,K,V,Z]. (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:582`) GDN forward splits as `[qkv_size, z_size]` = `[(key_dim*2+value_dim)/tp, value_dim/tp]`, comment says "weights already in [q,k,v,z] order". (`qwen_gdn_linear_attn.py:947-952`)
- **GDN forward non-zero** — `post_proj=1.9222` during 5-token inference.
- **embed_tokens single + multi-token** — cosine 0.999998 for BOS and Paris at n=4.
- **A_log, dt_bias** — F32 via `sharded_weight_loader`; `ssm_a ∈ [-72.33, -0.0186]`.
- **conv1d weight** — `blk.0.ssm_conv1d.weight` = 8192×4 = 32768 F32 elements.
- **is_layer_skipped_gguf Fix2** — `module_name in shard_prefix` direction correct; F32 layers get `UnquantizedLinearMethod`. (`vllm/model_executor/layers/quantization/gguf.py:142-163`)
- **TP=1 also fails** — garbage output on TP=1; bug is not TP-specific.
- **lm_head computation correct** — Manual dequantize + matmul: Paris logit=6.808, rank 1331. vLLM: Paris logit=6.8125, rank 1303. Cosine 0.999983. **Bug is in hidden states, not the logit projection.**
- **mRoPE positions correct for text-only** — Positions tensor is `[3, N]` with identical rows `[[0..N-1],[0..N-1],[0..N-1]]` for pure text inference. This is expected per `MRotaryEmbedding.get_next_input_positions_tensor`. (`vllm/model_executor/layers/rotary_embedding/mrope.py:401-414`)
- **MRotaryEmbedding triton kernel** — `t_mask` reads beyond `half_rd` at offsets ≥32 for interleaved mode, but q/k are loaded/stored only within `(offset < rd//2)` mask, so garbage reads have no effect. (`vllm/model_executor/layers/rotary_embedding/mrope.py:60-80`)
- **GDN HF state_dict shapes** — `in_proj_qkv [8192,2048]`, `in_proj_z [4096,2048]`, `in_proj_a [32,2048]`, `in_proj_b [32,2048]`. (`AutoModelForCausalLM.from_config` with `text_config`)
- **GDN GGUF tensors (blk.0)** — `attn_qkv [2048,8192]` (Q+K+V), `attn_gate [2048,4096]` (Z), `ssm_alpha [2048,32]`, `ssm_beta [2048,32]`, `ssm_conv1d [4,8192]`, `ssm_norm [128]`, `ssm_out [4096,2048]`.
- **Residuals grow per layer** — hidden state norm goes from ~1 (layer 0 input) to ~20 (layer 39 output); final norm output ~115. Layers ARE contributing to residual. (`/tmp/diag_layer_norms.py`)

---

## What has been ruled out

| ID | Hypothesis | Evidence |
|----|-----------|---------|
| H4 | GDN Triton kernel SM89/SM86 | RULED OUT: TP=1 fails |
| H5 | conv1d shape wrong for TP=2 | RULED OUT: TP=1 fails; shape correct |
| H6 | SSM state shape wrong for TP=2 | RULED OUT: TP=1 fails |
| H7 | conv1d zero-V | RULED OUT: 8192 full conv1d confirmed |
| H9 | `attn_metadata_raw is None` → GDN returns early | RULED OUT: `post_proj=1.9222` during inference |
| H_embed_multi | `ggml_dequantize` wrong for n_tokens>1 | RULED OUT: cosine 0.999998 at n=4 |
| H10 | blk.40 BF16 unmapped | BENIGN: blk.40 is speculative NEFT head only |
| H3 | `is_layer_skipped_gguf` namespace mismatch | FIXED by Fix2 + apply_vllm_mapper |
| H13 | `lm_head` weight not tied → uninitialized logits | RULED OUT: lm_head cosine 0.999983 vs reference |
| H12 | `norm.weight` wrong → GDN output scaled badly | RULED OUT: ssm_norm.weight verified non-trivial |
| H_mamba_ptr | MambaCopyBuffers int64 pointer overflow | FIXED: applied upstream `b5495cc5f`; `mamba_utils.py:242-243` now `torch.uint64` |
| H_mRoPE_text | mRoPE [3,N] positions wrong for text-only | RULED OUT: identical rows are correct; triton kernel handles it |

---

## Applied fixes (branch `dev_kjelloe`)

| ID | File | Description |
|----|------|-------------|
| A | `vllm/model_executor/model_loader/gguf_loader.py` | Add qwen3_5_moe dispatch block; map expert weight GGUF names |
| B | `vllm/model_executor/model_loader/gguf_loader.py` | Fix is_multimodal false-positive (Qwen3_5MoeConfig.vision_config uses `depth`) |
| C | `vllm/model_executor/model_loader/gguf_loader.py` | Apply is_multimodal fix in `_get_gguf_weight_type` and `_get_weights_iterator` |
| D(a-c) | `vllm/model_executor/model_loader/gguf_loader.py`, `config.py` | Wrap gguf checkpoint load; use text_config for dummy model |
| E | `vllm/model_executor/model_loader/gguf_loader.py` | Manually map `blk.N.ssm_dt.bias` → `dt_bias` |
| F | `vllm/model_executor/model_loader/gguf_loader.py` | Postprocess name_map to add `model.language_model.` prefix |
| G | `vllm/model_executor/layers/linear.py` | `MergedColumnParallelLinear`: add tuple shard_id GGUF path |
| H | `vllm/model_executor/models/qwen3_5.py` | Unsqueeze `shared_expert_gate` 1D→2D on load |
| I | `vllm/model_executor/models/qwen3_5_moe_mamba.py` (mamba_mixer2) | `mamba_v2_sharded_weight_loader`: unsqueeze 2D→3D for conv1d |
| J | `vllm/model_executor/layers/quantization/gguf.py` | Return zeros for `GGUFUninitializedParameter` in `apply()` |
| K | `vllm/model_executor/model_loader/gguf_loader.py` | Fix trailing-dot bug in `find_hf_name_in_tensor_map` |
| L | `vllm/model_executor/models/qwen3_5.py` | Pass `quant_config` to `VocabParallelEmbedding` |
| M | `vllm/model_executor/layers/quantization/gguf.py` | Fix imatrix MoE kernel dispatch (MMVQ→MMQ in `_fused_moe_gguf` elif) |
| Fix2 | `vllm/model_executor/layers/quantization/gguf.py` | `is_layer_skipped_gguf` wrong direction: `module_name in shard_prefix` |
| uint64 | `vllm/v1/worker/mamba_utils.py:242-243` | Cherry-pick `b5495cc5f`: MambaCopyBuffers src/dst ptrs `int64` → `uint64` |

---

## STRUCTURAL CHANGE (2026-07-04) — GGUF support migrated to an out-of-tree plugin

The upstream merge (`3e879ebc2`) pulled in PR #39612
"[Migration] Migrate GGUF quantization support to plugin", which **deleted the
entire in-tree GGUF subsystem** (`gguf_loader.py`, `quantization/gguf.py`,
`gguf_utils.py`, the `weight_utils` GGUF helpers, and the CUDA kernels) and moved
it to [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin)
(`uv pip install vllm-gguf-plugin`; installed `0.0.2`).

Consequence: **all Fixes A–M below live in files upstream no longer uses.** The
merge kept our modified `gguf_loader.py`/`gguf.py` as orphaned dead code (nothing
imports them; harmless, pending cleanup). GGUF is now provided solely by the
plugin's `OOTGGUFModelLoader` + a `weights_adapter` registry.

**The plugin does NOT support Qwen3.5(-MoE) yet.** Verified from its source
(`.venv/.../vllm_gguf_plugin/weights_adapter/default.py`):
- `build_name_map` looks up the gguf arch by matching HF `model_type` against
  `gguf.MODEL_ARCH_NAMES` values. Our config reports `model_type="qwen3_5_moe"`
  (text: `qwen3_5_moe_text`) but gguf-py's arch value is `"qwen35moe"` — no
  remap exists (it special-cases cohere/gemma3/qwen2_moe/qwen3_moe/olmoe/
  minimax_m2 only) → would raise `RuntimeError: Unknown gguf model_type`.
- `is_multimodal` is set from `config.vision_config is not None`; Qwen3.5's
  config carries a vision_config even for the text-only GGUF → false positive
  (same issue as old Fix B/C).
- No MoE expert sideload mapping for qwen3_5_moe (needs the `ffn_*_exps` block
  like qwen3_moe).
- No GDN V-head reorder (the root cause below is unfixed there too).

**Fix path (chosen: adopt plugin).** Add a `Qwen35GGUFAdapter(GGUFWeightsAdapter)`
to the plugin, registered in `weights_adapter/_ADAPTER_REGISTRY`, that:
1. remaps `qwen3_5_moe`/`qwen3_5_moe_text` → gguf `QWEN35MOE` (and dense →
   `QWEN35`);
2. forces text-only name mapping despite `vision_config`;
3. adds the MoE expert + shared-expert sideload entries;
4. overrides `transform_weight` to apply the GDN V-head tiling→grouped reorder
   (the root-cause fix — see below).
The plugin's `transform_weight(hf_name, weight)` hook receives the raw packed
GGUF tensor per weight (see `default.py:map_weights`), which is exactly the
weight-space site chosen for the reorder. This re-homes Fixes A–F into the
plugin's clean adapter API and adds the root-cause fix. Should be upstreamed as
"add Qwen3.5 support" to vllm-gguf-plugin.

**Duplicate-work check done (2026-07-04).** Plugin `main` has no qwen3_5 adapter
(only base/default/gemma3/diffusion). No *open* PR covers it. Closed PR **#31**
"Add Qwen3.5 and Gemma4 GGUF adapters" (AlexCheema `codex/qwen35-gemma4-adapters`)
DID implement it but was **bulk-closed by the maintainer (Isotr0py) today with no
comment** — part of a closed stack of bot-generated `codex/*` PRs, not a technical
rejection. Its `weights_adapter/qwen3_5.py` implements the **exact**
`_tiled_to_grouped_v_heads` inverse permutation independently derived above, on the
same tensor set — a second confirmation of the root cause.

**#31's quant strategy verified correct (empirically):** GGUF packed data for
`attn_qkv` is `(8192, 1680)` = `[out_features, packed_input_bytes]`, so permuting
rows along dim 0 moves whole independently-quantized output features — safe for the
row-reordered tensors (in_proj_qkv V-rows, in_proj_z, in_proj_a/b, conv1d V-part,
A_log, dt_bias). `out_proj` is `(2048, 4352)` with the reorder axis (input dim)
packed inside the bytes → #31 force-dequantizes only `out_proj`, which is correct.
Caveat: #31's only smoke test was Qwen3-0.6B (no GDN layers), so its GDN path was
never actually exercised — our 35B-A3B run is the real validation.

**Refined plan:** rather than transplant #31's 409-line `default.py` delta (bloated
with Gemma4 + vision-tower + MTP mappings), write a **self-contained**
`Qwen3_5GGUFAdapter` carrying its own text-only name-map (arch remap
`qwen3_5_moe`→`QWEN35MOE`, MoE + shared-expert sideload) plus the verified
`transform_weight`. Dev against the installed wheel `0.0.2` (editable install of the
clone fails — compiled `_C_gguf` extension), mirror the adapter to the local clone
`/home/kjelloe/GIT/vllm-gguf-plugin` for the PR. PR submission gated on user approval.

---

## ROOT CAUSE (confirmed 2026-07-04) — GDN V-head tiling mismatch

**Mechanism.** llama.cpp's converter reorders the GDN value heads from HF's
*grouped* layout to ggml's *tiled* layout for every arch where
`num_key_heads != num_value_heads`. Qwen3.5 qualifies (16 K-heads vs 32
V-heads, `num_v_per_k = 2`). vLLM's GDN implementation was written for HF
safetensors (grouped) and never undoes the tiling, so all 30 GDN layers read
V / Z-gate / beta / alpha / A_log / dt_bias / conv1d-V / out_proj heads in the
wrong order. Result: attention mixes the wrong value heads → context-word
copying, garbage output.

**Layouts** (`num_v_per_k = num_value_heads // num_key_heads = 2`):
- HF grouped (what vLLM expects): `[G0_v0, G0_v1, G1_v0, G1_v1, ...]`
  — V head `j` pairs with K head `j // 2`.
- ggml tiled (what the GGUF stores): `[G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]`
  — V head `j` pairs with K head `j % 16`.

**Evidence:**
1. Converter: `_LinearAttentionVReorderBase._reorder_v_heads` +
   `modify_tensors`, `llama.cpp/conversion/qwen.py:353-518`. `Qwen3_5MoeTextModel`
   (registered for `Qwen3_5MoeForCausalLM`) inherits it (`qwen.py:626`). Docstring:
   "reorders V heads from grouped to tiled order for ggml broadcast" (PR #19468).
   Reorders: V rows of `in_proj_qkv`; all of `in_proj_z`; `in_proj_a`/`in_proj_b`;
   `A_log`/`dt_bias`/`dt_proj`; V-channels of `conv1d`; columns of `out_proj`.
2. vLLM expects grouped: `fix_query_key_value_ordering`
   (`qwen_gdn_linear_attn.py:644-695`) reshapes the interleaved (Qwen3-Next) case
   as `(num_k_heads, [hk, hk, np/ng*hv, np/ng*hv])` → V heads emerge grouped by
   K-head. The Qwen3.5 path (`:946-952`) does a plain split and feeds V straight
   through in on-disk order. Grep confirms **no reorder/tiled/repeat_interleave**
   anywhere in `vllm/model_executor/layers/mamba/gdn/` nor in `gguf_loader.py`.
3. GGUF metadata: `head_count=16`, `ssm.inner_size=4096` (=32×128),
   `ssm.group_count=16` → num_v_per_k=2, tiling is a non-trivial permutation.
4. Symptom (context copying) is exactly what scrambled V/gate heads produce.

**This subsumes prior hypotheses:** H_gdn_qkv_order (the Q/K/V *blocks* are in
the right order — my earlier [K,Q,V] guess was wrong — but the V *heads within*
the block are permuted) and H14 (beta/alpha aren't swapped, their rows are
tiled-reordered — same mechanism).

**FIX — convert tiled → grouped (inverse of `_reorder_v_heads`).** For a tensor
with the V-head axis of length `num_v_heads*head_dim`, reshape that axis to
`(num_v_per_k, num_k_heads, head_dim)`, swap the first two → `(num_k_heads,
num_v_per_k, head_dim)`, flatten. Apply to the GGUF tensors listed above.
Permutation granularity is `head_v_dim=128`, a multiple of every quant block
size here (Q6_K/Q4_K super-block 256 along columns → rows independent; Q8_0
blocks of 32 → 128 = 4 blocks), so a **weight-space** permute at load is
quantization-safe. Two candidate fix sites (decision pending): (1) weight-space
at GGUF load — model becomes bit-identical to safetensors, forward untouched;
(2) activation-space in the GDN forward — no quant concerns, adds hot-path
index_selects. Both must be gated to GGUF-loaded Qwen3.5 only (safetensors are
already grouped).

---

## Plan of attack (order of execution)

Run these in order. Plans 0 and A are cross-cutting and cheap; they either
re-baseline or localize the bug before any per-hypothesis work.

### Plan 0 — Re-baseline after upstream merge (30 min)

The merge to upstream/main replaced the entire `load_weights` body with
`AutoWeightsLoader` + `WeightsMapper.orig_to_new_stacked`. Two things must be
re-confirmed before trusting any prior runtime measurement:

1. **Symptoms unchanged.** Run the Paris-rank test
   (`./debugging/run.sh debugging/test_tp1.py`). If output changed in any way
   (better or worse), the loading-path change is itself evidence — diff the
   loaded weights (step 2) before proceeding.
2. **Weights load identically.** The tuple-shard GGUF path (Fix G in
   `MergedColumnParallelLinear`) was exercised by the old manual loop calling
   `weight_loader(param, loaded_weight, shard_id)`. `AutoWeightsLoader` +
   mapper reaches the same loader through a different call chain
   (`WeightsMapper.apply` rewrites names, `_load_module` descends). Verify:
   dump `in_proj_qkvz.qweight` and `in_proj_ba.weight` checksums for layer 0
   on this branch vs the pre-merge commit (`git stash` / checkout the
   pre-merge SHA), compare. Also confirm no new
   "Parameter ... not found, skip loading" warnings appear in the load log —
   the old loop's GGUF `.weight`→`.qweight` fallback was dropped in the merge
   on the assumption the mapper handles it; this check proves it.

**Exit criterion:** identical symptoms + identical checksums → all prior
"confirmed OK" results remain valid. Otherwise stop and fix the load path
first.

### Plan A — Bisect first divergent layer vs llama.cpp (half day, highest information yield)

llama.cpp produces correct output from the *same file*, so it is a working
reference implementation. Instead of guessing which subsystem is broken,
find the first layer where activations diverge:

1. **Reference dump:** build llama.cpp with the `eval-callback` example
   (`llama-eval-callback -m <gguf> -p "The capital of France is"`); it prints
   every graph tensor (name, shape, sample values). Capture per-layer output:
   the tensors named `l_out-N` (residual stream after layer N) and the
   embedding output.
2. **vLLM dump:** extend `/tmp/diag_layer_norms.py` to also save the full
   hidden-state tensor per layer (first token is enough) to an `.npz`, not
   just norms.
3. **Compare:** cosine similarity per layer, first token. Expect cosine ≈1.0
   (within quantization noise, >0.99) up to some layer L, then a drop.
   - L = 0 (embedding diverges) → tokenizer/embedding path, unlikely given
     prior checks.
   - L is a GDN layer (0,1,2,4,5,6,...) → H_gdn_qkv_order or H14; go to
     their plans with high confidence.
   - L is a full-attention layer (3,7,11,...) → H16 (and check the
     attn_output_gate split); H_gdn_qkv_order is then effectively ruled out.
   - Divergence only appears at token 2+ (decode) but prefill matches →
     state management (H11 / conv1d state / KV cache), not weights.
4. **Caveat:** dtype differences (llama.cpp F32 graph vs vLLM BF16 activations)
   mean cosine won't be exactly 1.0; look for the *cliff*, not the absolute
   value.

**Exit criterion:** a specific layer index and layer type where divergence
starts. This converts every hypothesis below from "possible" to
"confirmed/ruled out" in one run.

---

## Active hypotheses (ranked by priority)

### H_gdn_qkv_order — GGUF attn_qkv internal ordering ≠ [Q,K,V] (HIGHEST)

**Root cause:** The GGUF file stores `blk.N.attn_qkv.weight` (shape [2048, 8192]) with Q+K+V concatenated. The vLLM code assumes this is [Q, K, V] order (comment: "weights already in [q, k, v, z] order", `qwen_gdn_linear_attn.py:947`). But llama.cpp typically concatenates as [K, Q, V] or [K, V, Q] for some architectures. If the actual GGUF order is different from [Q, K, V], the GDN recurrence will use wrong query/key/value assignments in all 30 GDN layers.

**Why plausible:** Context-word copying is exactly what you'd expect if attention keys and queries are swapped (all tokens look identical to each other, attention collapses to positionally-biased averaging → echoes most recent context token).

**Diagnosis:**
```python
# Load in_proj_qkv weight directly from GGUF and compare first 2048
# columns (should be Q) with what vLLM loaded as the Q portion of in_proj_qkvz
from gguf import GGUFReader
import os, torch
r = GGUFReader(os.path.join(os.environ['LLAMA_MODELS_DIR'],
               'Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf'))
t = next(t for t in r.tensors if t.name == 'blk.0.attn_qkv.weight')
raw = torch.tensor(t.data)
# Compare first 2048 rows (Q?) vs rows 2048:4096 (K?) vs rows 4096:8192 (V?)
# Check against what llama.cpp does for Qwen3.5 ssm layers
# File: https://github.com/ggerganov/llama.cpp/blob/master/src/llama-arch.cpp
# Look for QWEN35MOE or similar block definition
```

**File to check:** llama.cpp `src/llama-arch.cpp` — find the qkv ordering for GDN/SSM layers in Qwen3.5 (search for `LLM_TENSOR_ATTN_QKV` and `QWEN`). Also check `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:947-952`.

**PLAN:**

*Step 1 — Establish ground truth from the converter (authoritative, no GPU needed).*
The GGUF file was produced by llama.cpp's `convert_hf_to_gguf.py`. Whatever
concatenation order that script used when fusing the HF `in_proj_qkv` (or
separate q/k/v tensors) into `blk.N.attn_qkv` **is** the on-disk order. Read:
- `convert_hf_to_gguf.py` — the model class handling Qwen3.5 (search
  `qwen3_5`, `Qwen35`, or the GDN tensor names); look at `modify_tensors` for
  how `attn_qkv` is assembled.
- `src/llama-model.cpp` (build graph) — how llama.cpp *splits* `attn_qkv` at
  inference for the GDN layers. The split order there must match the concat
  order in the converter; that pair defines ground truth.
- Cross-check the HF side: does HF `in_proj_qkv` itself store [Q,K,V]
  contiguously? (Check `modeling_qwen3_5` in transformers — the `.split()`
  call in the HF GDN forward.)

*Step 2 — Empirical confirmation (only if step 1 is ambiguous).*
Sizes are Q=2048, K=2048, V=4096 rows, so a Q↔K swap is invisible to shape
checks — but V's placement is not. Dequantize `blk.0.attn_qkv` rows
[0:2048], [2048:4096], [4096:8192] and compare per-segment row-norm
distributions against the runtime `in_proj_qkvz` shards. If V (4096 rows) is
not in the last position, that is detectable purely from shapes of the
matching segments.

*Step 3 — Fix design (if order differs).*
Do **not** touch the GDN forward split (`qwen_gdn_linear_attn.py:947-952`) —
it is shared with safetensors loading which works. Fix at GGUF load time by
row-permuting the raw tensor before shard placement:
- Q*_K quantization blocks are per-row (each row quantizes independently
  along the input dim), so permuting rows of the qweight byte tensor is safe:
  view as `[8192, row_bytes]`, index-select segments into [Q,K,V] order.
- Placement: the `_preprocess` generator already in
  `Qwen3_5Model.load_weights` (`qwen3_5.py:271-276`) is the natural hook —
  match `name.endswith("in_proj_qkv.qweight")`, permute, yield. Gate it on
  GGUF (weight is a qweight byte tensor) so safetensors is untouched.
- Same treatment may be needed for `attn_qkv` bias if present (it is not, per
  tensor dump — weight only).

*Step 4 — Validate.* Paris rank test; expect rank ≈1. Then layer-norm diag
to confirm hidden-state norms look sane, then a 50-token coherence sample.

**Effort:** step 1 ~1h reading; fix ~1h; low risk.

---

### H14 — `in_proj_ba` shard values wrong → GDN recurrence saturated (MEDIUM)

**Root cause:** `in_proj_ba` is `MergedColumnParallelLinear` with `output_sizes=[num_v_heads, num_v_heads]` = [b_size, a_size]. If `in_proj_b` and `in_proj_a` are loaded with swapped shard ids (b→shard1, a→shard0 instead of b→shard0, a→shard1), the decay/input-mixing coefficients are swapped. Post-merge mapping: `".in_proj_b": (".in_proj_ba", 0)`, `".in_proj_a": (".in_proj_ba", 1)` in `hf_to_vllm_mapper` (`vllm/model_executor/models/qwen3_5.py:203-210`). There is a second swap opportunity upstream of that: the GGUF name map decides whether `ssm_beta`→`in_proj_b` and `ssm_alpha`→`in_proj_a` or vice versa.

**Diagnosis:** Run `/tmp/diag_runtime_weights.py` — checks b/a range for layer 0 at runtime. `ssm_beta` (input mixing) and `ssm_alpha` (decay) should have very different value ranges.

**PLAN:**

*Step 1 — Offline exact comparison (no engine needed, avoids the OOM that
blocked the last attempt).* Both `ssm_alpha` and `ssm_beta` are F32 in the
GGUF, so runtime values must match the file bit-exactly. Rather than booting
the whole LLM engine, write a script that:
1. Reads `blk.0.ssm_alpha.weight` and `blk.0.ssm_beta.weight` via
   `gguf.GGUFReader` (CPU, seconds).
2. Reads the GGUF→HF name map by calling
   `GGUFModelLoader._get_gguf_weights_map` equivalent logic, or simply
   greps the two names through `gguf.get_tensor_name_map(QWEN35MOE, 40)`,
   to confirm which HF name each maps to.
3. Cross-checks against the HF reference: in `transformers`
   `modeling_qwen3_5`, which of `in_proj_b`/`in_proj_a` feeds the
   `sigmoid` (β, input gate) and which feeds `-softplus`-style decay (a).
   Then check llama.cpp's GDN graph uses `ssm_beta`/`ssm_alpha` the same way.

*Step 2 — If the full-engine check is still wanted:* rerun
`/tmp/diag_runtime_weights.py` with `gpu_memory_utilization=0.90` and
`max_model_len=128` (the previous failure was 0.08 GiB short at
`max_model_len=256`).

*Step 3 — Fix design (if swapped).* Fix at the name-map level in
`gguf_loader.py` (the qwen3_5_moe dispatch block from Fix A/E): map
`ssm_alpha` ↔ `ssm_beta` to the correct `in_proj_a`/`in_proj_b` HF names.
Do not change the shard ids in `hf_to_vllm_mapper` — those must stay aligned
with the safetensors layout.

*Step 4 — Validate.* Paris rank test. Note: a b/a swap alone corrupts the
recurrence gain/decay but might not fully explain context-copying; if output
improves but is still wrong, return to Plan A.

**Effort:** step 1 ~30 min; fix trivial; low risk.

---

### H16 — KV cache layout wrong for full-attention layers (MEDIUM)

**Root cause:** Full-attention layers (3, 7, 11, ..., 39) use standard KV cache. For this hybrid model the KV cache block might have wrong head count or head dimension, causing corrupted KV retrieval. `Qwen3NextAttention` has `num_heads=16`, `num_kv_heads=2`, `head_dim=256`.

**File:** `vllm/model_executor/models/qwen3_next.py:267-281` (Attention layer construction).

**Diagnosis:** Add a hook inside `Qwen3NextAttention.forward` to print the actual attention output norm vs the expected. If attention output is near-zero or near-uniform the KV cache is broken.

**PLAN:**

*Sub-hypothesis H16b (check FIRST — new, cheap, and plausible): Q/gate
interleave mismatch in `attn_q`.* The eager split in
`Qwen3NextAttention._project_qkv_gate` (`qwen3_next.py:359-367`) views the
8192-dim q_gate as `[num_heads=16, 2*head_dim=512]` and chunks per head —
i.e., it assumes the weight rows are laid out **per-head interleaved**:
`[q_h0(256) | gate_h0(256) | q_h1 | gate_h1 | ...]`. The prior "cosine
0.999984" check only proved vLLM loaded the GGUF bytes faithfully; it says
nothing about whether the GGUF's `attn_q` row layout matches this interleave
assumption. If llama.cpp's converter stores `[all Q (4096) | all gates
(4096)]` (or applies its own permutation), every full-attention layer
computes attention with half-gate/half-Q garbage and gates the output with
sigmoid(noise).
1. Read `convert_hf_to_gguf.py` for Qwen3.5 `attn_q` handling — is HF
   `q_proj` copied verbatim or rearranged?
2. Read llama.cpp's qwen3.5 attention graph — how does it split Q vs gate
   from `attn_q`? Its split convention reveals the storage convention.
3. If mismatched: fix by row-permuting `attn_q` at GGUF load time (same
   `_preprocess` hook and same per-row-quantization-safety argument as
   H_gdn_qkv_order step 3).

*KV-cache check proper (if H16b is clean):*
1. Confirm layer registration: dump `kv_cache_spec` from the model runner
   for layers 3,7,...,39; verify `num_kv_heads=2`, `head_size=256`, and that
   the hybrid manager gave full-attention layers `FullAttentionSpec` (not a
   mamba spec) with consistent `block_size`.
2. Prefill-only test: 1 prompt, `max_tokens=1`. Prefill attention reads K/V
   from the current batch, mostly bypassing paged-cache reads. If the very
   first generated token is already garbage (it is, per symptoms), the bug is
   in prefill compute — weights/layout — and *not* in paged KV-cache
   bookkeeping. That would demote the cache-layout half of H16 to LOW and
   leave H16b standing.
3. Only if decode diverges while prefill matches (see Plan A step 3): dump
   K/V written to cache during prefill for layer 3 and re-read them at
   decode step 1; compare bytes.

**Effort:** H16b ~1h reading + trivial fix; cache checks ~2h.

---

### H11 — GDN SSM state slot indices wrong → stale recurrent state (LOW)

**Root cause:** GDN layers index into a shared state tensor pool by layer name. If the index lookup is wrong (e.g., name collision, off-by-one), two layers might share state or read zeros.

**File:** `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:_forward_core` — looks up state via `compilation_config.static_forward_context`.

**PLAN:**

*Step 1 — Cheap logical elimination.* The symptom is garbage on the *first*
generated token of a fresh 5-token prompt. During that prefill, the
recurrent state starts from zeros and is built within the chunked-scan
kernel itself; cross-step state plumbing barely participates. If Plan A
shows prefill already diverges, H11 is ruled out without further work.

*Step 2 — Direct check (only if decode-onset divergence).* One-time hook in
`_forward_core`: log `id(ssm_state)` / state-tensor slot index per layer for
all 30 GDN layers over two decode steps; assert 30 unique slots and that
layer N reads at step t what it wrote at step t-1 (checksum a small slice).

*Step 3 — Fix design.* If a collision exists it will be in the
layer-name→slot mapping built from `static_forward_context`; fix the key
derivation (likely a prefix-normalization issue from the
`model.language_model.` remapping, Fix F territory). No kernel changes.

**Effort:** step 1 free (falls out of Plan A); step 2 ~1h.

---

## Diagnostic scripts

```bash
# Runtime weight value check (H14 / H_gdn_qkv_order sanity)
./debugging/run.sh /tmp/diag_runtime_weights.py

# Layer-by-layer hidden state norms
./debugging/run.sh /tmp/diag_layer_norms.py

# TP=1 coherence test
./debugging/run.sh debugging/test_tp1.py
```

---

## Key architecture facts (for reference)

**Model:** Qwen3.5-35B-A3B Q4_K_M GGUF, 40 layers (3 GDN + 1 full_attn × 10)  
**Config source:** `Qwen/Qwen3.5-35B-A3B` HF config via `hf_config_path`

| Config key | Value |
|---|---|
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| attn_output_gate | True (Q proj = 32 heads × 256 = 8192 dims, includes gate) |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 32 |
| linear_key_head_dim | 128 |
| linear_value_head_dim | 128 |
| gqa_interleaved_layout | False (Qwen3.5 non-interleaved) |
| rope_type | default (text-only mRoPE) |
| mrope_section | [11, 11, 10] |
| mrope_interleaved | True |
| partial_rotary_factor | 0.25 → rotary_dim=64 |
| rope_theta | 10000000 |

**GDN key dims:**
- `key_dim` = 16 × 128 = 2048 (Q and K size)
- `value_dim` = 32 × 128 = 4096 (V and Z size)
- `in_proj_qkvz` output_sizes = [2048, 2048, 4096, 4096] = [Q, K, V, Z]
- `in_proj_ba` output_sizes = [32, 32] = [b, a]

**Weight loading chain for GDN:**
1. `gguf_to_hf_name_map`: `blk.N.attn_qkv.weight` → `model.layers.N.linear_attn.in_proj_qkv.qweight`
2. `gguf_to_hf_name_map`: `blk.N.attn_gate.weight` → `model.layers.N.linear_attn.in_proj_z.qweight`
3. `hf_to_vllm_mapper.orig_to_new_stacked`: `in_proj_qkv` → `in_proj_qkvz` shard `(0,1,2)` (Q+K+V block)
4. `hf_to_vllm_mapper.orig_to_new_stacked`: `in_proj_z` → `in_proj_qkvz` shard `3` (Z)
5. MergedColumnParallelLinear places shards at output offsets per `output_sizes=[2048,2048,4096,4096]`
6. GDN forward splits: `mixed_qkvz → [qkv_size=8192 | z_size=4096]` (`qwen_gdn_linear_attn.py:948-950`)

**Relevant files:**
- `vllm/model_executor/model_loader/gguf_loader.py` — GGUF→HF name mapping (`_get_gguf_weights_map`)
- `vllm/model_executor/model_loader/weight_utils.py:1281` — `gguf_quant_weights_iterator`
- `vllm/model_executor/models/qwen3_5.py:203-210` — `hf_to_vllm_mapper.orig_to_new_stacked` (replaces the old `stacked_params_mapping` after upstream merge); `load_weights` with GGUF `_preprocess` hook at `:252-279`
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — GDN forward, weight construction
- `vllm/model_executor/layers/rotary_embedding/mrope.py:201` — `MRotaryEmbedding`
- `vllm/model_executor/models/qwen3_next.py:207` — `Qwen3NextAttention`
- `vllm/v1/worker/mamba_utils.py:242-243` — MambaCopyBuffers (uint64 fix applied)
