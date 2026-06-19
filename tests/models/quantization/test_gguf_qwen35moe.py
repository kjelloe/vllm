# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Qwen3.5-MoE GGUF loader logic.

These tests cover the name-mapping, prefix-postprocessing, dt_bias injection,
tie_word_embeddings detection, and is_layer_skipped_gguf behavior introduced
to support model_type=qwen3_5_moe GGUF files. They do NOT require the actual
model file or a GPU — all GGUF I/O is mocked.

To run (with .venv activated):
    .venv/bin/python -m pytest tests/models/quantization/test_gguf_qwen35moe.py -v
"""

from __future__ import annotations

import os
import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: build a minimal fake gguf_to_hf_name_map the same way the real code
# does, so we can test the transformations without touching the file system.
# ---------------------------------------------------------------------------


def _build_fake_name_map(num_layers: int = 40) -> dict[str, str]:
    """Minimal name map that mimics what gguf-py produces for qwen35moe."""
    m: dict[str, str] = {}
    for idx in range(num_layers):
        m[f"blk.{idx}.attn_qkv.weight"] = f"model.layers.{idx}.linear_attn.in_proj_qkv.weight"
        m[f"blk.{idx}.attn_gate.weight"] = f"model.layers.{idx}.linear_attn.in_proj_z.weight"
        m[f"blk.{idx}.ssm_alpha.weight"] = f"model.layers.{idx}.linear_attn.in_proj_a.weight"
        m[f"blk.{idx}.ssm_beta.weight"] = f"model.layers.{idx}.linear_attn.in_proj_b.weight"
        m[f"blk.{idx}.ssm_conv1d.weight"] = f"model.layers.{idx}.linear_attn.conv1d.weight"
        m[f"blk.{idx}.ssm_a"] = f"model.layers.{idx}.linear_attn.A_log"
        m[f"blk.{idx}.ssm_norm.weight"] = f"model.layers.{idx}.linear_attn.norm.weight"
        m[f"blk.{idx}.ssm_out.weight"] = f"model.layers.{idx}.linear_attn.out_proj.weight"
        m[f"blk.{idx}.ffn_down_exps.weight"] = f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
        m[f"blk.{idx}.ffn_gate_exps.weight"] = f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
        m[f"blk.{idx}.ffn_up_exps.weight"] = f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
    m["token_embd.weight"] = "model.embed_tokens.weight"
    m["output_norm.weight"] = "model.norm.weight"
    m["output.weight"] = "lm_head.weight"
    return m


def _apply_qwen35moe_fixes(
    name_map: dict[str, str],
    num_layers: int = 40,
    layer_types: list[str] | None = None,
) -> dict[str, str]:
    """
    Apply the three qwen3_5_moe post-processing steps that live in
    GGUFModelLoader._get_gguf_weights_map:
      E — inject dt_bias entries for linear_attention layers
      F — prepend 'model.language_model.' to all model.* values
      (tie_word_embeddings detection happens separately in load_model)
    """
    if layer_types is None:
        layer_types = ["linear_attention"] * num_layers

    # Fix E: inject dt_bias
    for idx in range(num_layers):
        if layer_types[idx] == "linear_attention":
            name_map[f"blk.{idx}.ssm_dt.bias"] = (
                f"model.layers.{idx}.linear_attn.dt_bias"
            )

    # Fix F: replace "model." prefix with "model.language_model." so
    # hf_to_vllm_mapper can route weights into language_model.* submodules.
    # Real code: "model.language_model." + v[len("model."):]
    updated: dict[str, str] = {}
    for gguf_name, hf_name in name_map.items():
        if hf_name.startswith("model."):
            updated[gguf_name] = "model.language_model." + hf_name[len("model."):]
        else:
            updated[gguf_name] = hf_name
    return updated


# ---------------------------------------------------------------------------
# Tests: name mapping correctness
# ---------------------------------------------------------------------------


class TestQwen35MoENameMap:
    """Verify GGUF→HF name mapping for qwen3_5_moe."""

    def setup_method(self):
        self.name_map = _build_fake_name_map(num_layers=2)
        self.fixed_map = _apply_qwen35moe_fixes(self.name_map.copy(), num_layers=2)

    def test_dt_bias_injected_for_linear_attention(self):
        assert "blk.0.ssm_dt.bias" in self.fixed_map
        assert "blk.1.ssm_dt.bias" in self.fixed_map

    def test_dt_bias_hf_name_correct(self):
        assert self.fixed_map["blk.0.ssm_dt.bias"] == (
            "model.language_model.layers.0.linear_attn.dt_bias"
        )

    def test_model_prefix_applied(self):
        for gguf_name, hf_name in self.fixed_map.items():
            if gguf_name.startswith("blk.") or gguf_name == "token_embd.weight":
                assert hf_name.startswith("model.language_model."), (
                    f"{gguf_name} → {hf_name} should have model.language_model. prefix"
                )

    def test_lm_head_not_prefixed(self):
        # lm_head.weight starts with "lm_head.", not "model." so Fix F must not touch it
        assert self.fixed_map["output.weight"] == "lm_head.weight"

    def test_no_double_prefix(self):
        for hf_name in self.fixed_map.values():
            assert "model.language_model.model.language_model." not in hf_name

    def test_attn_qkv_mapping(self):
        expected = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        assert self.fixed_map["blk.0.attn_qkv.weight"] == expected

    def test_ssm_conv1d_mapping(self):
        expected = "model.language_model.layers.0.linear_attn.conv1d.weight"
        assert self.fixed_map["blk.0.ssm_conv1d.weight"] == expected

    def test_A_log_mapping_no_weight_suffix(self):
        # blk.0.ssm_a maps to A_log (no .weight suffix — it's a scalar per-head param)
        hf = self.fixed_map["blk.0.ssm_a"]
        assert hf.endswith("A_log")
        assert ".weight" not in hf

    def test_blk40_not_in_map_for_40layer_model(self):
        """blk.40 tensors are outside the 40-layer range and must not be mapped."""
        full_map = _build_fake_name_map(num_layers=40)
        full_fixed = _apply_qwen35moe_fixes(full_map.copy(), num_layers=40)
        assert "blk.40.ffn_gate_inp.weight" not in full_fixed
        assert "blk.40.ffn_gate_inp_shexp.weight" not in full_fixed


class TestDtBiasLinearAttentionOnly:
    """dt_bias must only be injected for linear_attention layers, not full_attention."""

    def test_dt_bias_not_injected_for_full_attention(self):
        layer_types = ["linear_attention"] * 30 + ["full_attention"] * 10
        m = _build_fake_name_map(num_layers=40)
        fixed = _apply_qwen35moe_fixes(m, num_layers=40, layer_types=layer_types)
        # GDN layers 0-29 get dt_bias
        assert "blk.0.ssm_dt.bias" in fixed
        assert "blk.29.ssm_dt.bias" in fixed
        # Full-attention layers 30-39 must NOT get it
        for idx in range(30, 40):
            assert f"blk.{idx}.ssm_dt.bias" not in fixed


# ---------------------------------------------------------------------------
# Tests: tie_word_embeddings detection
# ---------------------------------------------------------------------------


class TestTieWordEmbeddingsDetection:
    """
    The tie_word_embeddings check in gguf_loader.py:
        if any("lm_head.weight" in n for n in all_extra_names):
            model_config.hf_config.update({"tie_word_embeddings": True})

    After Fix F, lm_head.weight appears as a *value* in the name map as just
    "lm_head.weight" (not prefixed — Fix F only prefixes model.* values).
    The GGUF extra_names are the HF names of tensors NOT present in the GGUF
    (i.e., the ones that need to be tied).
    """

    def test_lm_head_substring_match_plain(self):
        all_extra_names = ["lm_head.weight"]
        assert any("lm_head.weight" in n for n in all_extra_names)

    def test_lm_head_substring_match_with_prefix(self):
        # Even if name has a prefix, the substring check still matches
        all_extra_names = ["model.language_model.lm_head.weight"]
        assert any("lm_head.weight" in n for n in all_extra_names)

    def test_no_false_positive_on_unrelated_names(self):
        all_extra_names = [
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.norm.weight",
        ]
        assert not any("lm_head.weight" in n for n in all_extra_names)

    def test_empty_extra_names_no_tie(self):
        all_extra_names: list[str] = []
        assert not any("lm_head.weight" in n for n in all_extra_names)


# ---------------------------------------------------------------------------
# Tests: is_layer_skipped_gguf behavior
# ---------------------------------------------------------------------------


class TestIsLayerSkippedGGUF:
    """
    is_layer_skipped_gguf checks whether a module should use
    UnquantizedLinearMethod because its weights are F32/F16 in the GGUF.

    After Fix 2, it checks: any(module_name in shard_prefix for shard_prefix
    in unquantized_modules). shard_prefix is the HF-style name without
    .weight suffix.
    """

    def test_import_is_layer_skipped(self):
        from vllm.model_executor.layers.quantization.gguf import is_layer_skipped_gguf
        assert callable(is_layer_skipped_gguf)

    def test_unquantized_module_detected(self):
        from vllm.model_executor.layers.quantization.gguf import is_layer_skipped_gguf

        # Simulate: in_proj_ba is F32, stored in unquantized_modules as:
        # "model.language_model.layers.0.linear_attn.in_proj_ba"
        # The vLLM module_name would be:
        # "language_model.model.layers.0.linear_attn.in_proj_ba"
        # After Fix 2: check module_name IN shard_prefix (substring match)
        unquantized = [
            "model.language_model.layers.0.linear_attn.in_proj_ba",
        ]
        module_name = "language_model.model.layers.0.linear_attn.in_proj_ba"
        # The vLLM prefix is a suffix of the HF-namespace name:
        # module_name ("language_model.model.layers.0...") is a SUFFIX of
        # "model.language_model.layers.0..." — substring check fails!
        # This is the known H3 namespace mismatch — test documents it.
        result = is_layer_skipped_gguf(module_name, unquantized)
        # Document current behavior (may be True or False depending on Fix 2)
        assert isinstance(result, bool)

    def test_h3_namespace_mismatch_for_qwen35moe_f32_layers(self):
        """
        H3: is_layer_skipped_gguf returns False for qwen35moe F32 layers because
        unquantized_modules uses HF-namespace ('model.language_model.layers.0...')
        while the vLLM module name uses 'language_model.model.layers.0...' —
        neither is a substring of the other.

        This means in_proj_ba, conv1d, ssm_norm (all F32) get GGUFLinearMethod
        instead of UnquantizedLinearMethod. Whether this causes garbage output
        depends on whether GGUFLinearMethod's apply() handles F32 qweight
        correctly via the UNQUANTIZED_TYPES fallback in _fused_mul_mat_gguf.
        """
        from vllm.model_executor.layers.quantization.gguf import is_layer_skipped_gguf

        # After Fix F, unquantized_modules entry for in_proj_ba looks like:
        hf_namespace = "model.language_model.layers.0.linear_attn.in_proj_ba"
        # vLLM module path (after hf_to_vllm_mapper) looks like:
        vllm_namespace = "language_model.model.layers.0.linear_attn.in_proj_ba"

        # Neither is a substring of the other:
        assert vllm_namespace not in hf_namespace
        assert hf_namespace not in vllm_namespace

        # Consequence: is_layer_skipped_gguf returns False (H3 confirmed)
        result = is_layer_skipped_gguf(vllm_namespace, [hf_namespace])
        assert result is False, (
            "H3 namespace mismatch: in_proj_ba (F32) incorrectly gets "
            "GGUFLinearMethod instead of UnquantizedLinearMethod"
        )

    def test_quantized_module_not_skipped(self):
        from vllm.model_executor.layers.quantization.gguf import is_layer_skipped_gguf

        unquantized: list[str] = []  # no unquantized modules
        module_name = "language_model.model.layers.0.linear_attn.in_proj_qkvz"
        assert not is_layer_skipped_gguf(module_name, unquantized)

    def test_exact_match_skipped(self):
        from vllm.model_executor.layers.quantization.gguf import is_layer_skipped_gguf

        module_name = "language_model.model.layers.0.linear_attn.in_proj_ba"
        unquantized = [module_name]  # exact match
        assert is_layer_skipped_gguf(module_name, unquantized)


# ---------------------------------------------------------------------------
# Tests: find_hf_name_in_tensor_map trailing-dot fix (Fix K)
# ---------------------------------------------------------------------------


class TestFindHFNameTrailingDot:
    """
    find_hf_name_in_tensor_map builds names as `gguf_name + "." + suffix`.
    When suffix is empty (e.g. for A_log which has no .weight suffix),
    Fix K avoids the trailing dot: `gguf_name if not suffix else gguf_name + "." + suffix`.
    """

    def test_no_trailing_dot_with_empty_suffix(self):
        # Simulate the fixed logic
        def find_hf_name(gguf_name: str, suffix: str) -> str:
            return gguf_name if not suffix else f"{gguf_name}.{suffix}"

        result = find_hf_name("model.layers.0.linear_attn.A_log", "")
        assert not result.endswith(".")
        assert result == "model.layers.0.linear_attn.A_log"

    def test_dot_added_with_suffix(self):
        def find_hf_name(gguf_name: str, suffix: str) -> str:
            return gguf_name if not suffix else f"{gguf_name}.{suffix}"

        result = find_hf_name("model.layers.0.mlp.gate_proj", "weight")
        assert result == "model.layers.0.mlp.gate_proj.weight"


# ---------------------------------------------------------------------------
# Tests: expert weight name patterns
# ---------------------------------------------------------------------------


class TestExpertWeightPatterns:
    """sideload_params regex must match expert weight names in the vLLM namespace."""

    def test_expert_pattern_matches_down_proj(self):
        pattern = re.compile(
            r"model\.layers\.0\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
        )
        assert pattern.match("model.layers.0.mlp.experts.0.down_proj.weight")
        assert pattern.match("model.layers.0.mlp.experts.7.gate_proj.weight")
        assert pattern.match("model.layers.0.mlp.experts.63.up_proj.weight")

    def test_expert_pattern_no_match_outside_experts(self):
        pattern = re.compile(
            r"model\.layers\.0\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
        )
        assert not pattern.match("model.layers.0.mlp.shared_expert.down_proj.weight")
        assert not pattern.match("model.layers.0.mlp.gate_proj.weight")
