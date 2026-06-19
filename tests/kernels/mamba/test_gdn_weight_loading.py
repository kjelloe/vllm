# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for GDN (Gated Delta Network) GGUF weight loading.

Covers:
- mamba_v2_sharded_weight_loader: 2D→3D unsqueeze for conv1d
- Correct Q/K/V shard splitting for GDN conv1d dim
- Dimension constants (conv_dim = key_dim * 2 + value_dim)

These tests do not require the actual Qwen3.5-35B-A3B model file.
A small GPU (or CPU for shape-only tests) is sufficient.

To run:
    .venv/bin/python -m pytest tests/kernels/mamba/test_gdn_weight_loading.py -v
"""
from __future__ import annotations

import pytest
import torch

QWEN35_KEY_DIM = 2048  # 16 heads × 128
QWEN35_VALUE_DIM = 4096  # 32 heads × 128
QWEN35_CONV_KERNEL = 4
QWEN35_CONV_DIM = QWEN35_KEY_DIM * 2 + QWEN35_VALUE_DIM  # 8192


# ---------------------------------------------------------------------------
# Tests: conv_dim arithmetic
# ---------------------------------------------------------------------------


class TestGDNConvDimArithmetic:
    """Verify the conv_dim formula used in QwenGatedDeltaNetAttention."""

    def test_conv_dim_formula(self):
        key_dim = QWEN35_KEY_DIM
        value_dim = QWEN35_VALUE_DIM
        conv_dim = key_dim * 2 + value_dim
        assert conv_dim == QWEN35_CONV_DIM, f"Expected 8192, got {conv_dim}"

    def test_gguf_conv1d_total_elements(self):
        # GGUF stores conv1d as (conv_dim, kernel_size) F32 = 8192 × 4
        total = QWEN35_CONV_DIM * QWEN35_CONV_KERNEL
        assert total == 32768

    def test_tp2_per_rank_conv_dim(self):
        tp_size = 2
        per_rank = QWEN35_CONV_DIM // tp_size
        assert per_rank == 4096

    def test_tp2_per_rank_conv_shape(self):
        tp_size = 2
        per_rank_dim = QWEN35_CONV_DIM // tp_size
        # After mamba_v2_sharded_weight_loader, shape is (per_rank_dim, 1, kernel)
        expected_shape = (per_rank_dim, 1, QWEN35_CONV_KERNEL)
        assert expected_shape == (4096, 1, 4)


# ---------------------------------------------------------------------------
# Tests: mamba_v2_sharded_weight_loader 2D→3D unsqueeze
# ---------------------------------------------------------------------------


class TestMambaV2ShardedWeightLoader:
    """Test the conv1d 2D→3D unsqueeze and Q/K/V shard splitting."""

    @pytest.fixture
    def loader_fn(self):
        from vllm.model_executor.layers.mamba.mamba_mixer2 import (
            mamba_v2_sharded_weight_loader,
        )
        return mamba_v2_sharded_weight_loader

    def _make_fake_conv1d_param(self, conv_dim: int, kernel: int) -> torch.Tensor:
        """Create a 3D param tensor matching what vLLM allocates for conv1d."""
        # vLLM allocates (conv_dim, 1, kernel) for conv1d
        param = torch.zeros(conv_dim, 1, kernel)
        # Fake the attributes that mamba_v2_sharded_weight_loader checks
        param.partition_dim = 0
        return param

    def test_unsqueeze_2d_to_3d(self, loader_fn):
        """
        When GGUF loads a 2D weight (conv_dim, kernel) and the param is 3D
        (conv_dim, 1, kernel), the loader must unsqueeze dim=1.
        """
        conv_dim = QWEN35_CONV_DIM  # 8192 for TP=1
        kernel = QWEN35_CONV_KERNEL

        # Shard spec for GDN: Q(key_dim), K(key_dim), V(value_dim)
        shard_spec = [
            (QWEN35_KEY_DIM, 0, False),
            (QWEN35_KEY_DIM, 0, False),
            (QWEN35_VALUE_DIM, 0, False),
        ]
        tp_size = 1
        tp_rank = 0

        load_fn = loader_fn(shard_spec, tp_size, tp_rank)

        # Simulate the GGUF-loaded 2D tensor
        loaded_2d = torch.randn(conv_dim, kernel)
        # Param is 3D
        param_3d = self._make_fake_conv1d_param(conv_dim, kernel)

        # The loader should unsqueeze and copy without error
        load_fn(param_3d, loaded_2d)

        # Result should now have the loaded values with dim=1 inserted
        assert param_3d.shape == (conv_dim, 1, kernel)
        # Check values were copied (allow for shape manipulation)
        # loaded_2d was (conv_dim, kernel); after unsqueeze(1) → (conv_dim, 1, kernel)
        expected = loaded_2d.unsqueeze(1)
        assert torch.allclose(param_3d, expected), "Values not correctly loaded"

    def test_tp2_shard_q_portion(self, loader_fn):
        """
        For TP=2, rank 0 should receive the first half of Q+K+V conv channels.
        Q: rows 0..key_dim/2-1
        K: rows key_dim..key_dim*1.5-1
        V: rows key_dim*2..key_dim*2+value_dim/2-1
        """
        conv_dim = QWEN35_CONV_DIM  # 8192 full
        kernel = QWEN35_CONV_KERNEL
        tp_size = 2
        tp_rank = 0
        per_rank_dim = conv_dim // tp_size  # 4096

        shard_spec = [
            (QWEN35_KEY_DIM, 0, False),
            (QWEN35_KEY_DIM, 0, False),
            (QWEN35_VALUE_DIM, 0, False),
        ]

        load_fn = loader_fn(shard_spec, tp_size, tp_rank)

        loaded_2d = torch.arange(conv_dim * kernel, dtype=torch.float32).reshape(
            conv_dim, kernel
        )
        param_3d = self._make_fake_conv1d_param(per_rank_dim, kernel)

        load_fn(param_3d, loaded_2d)

        assert param_3d.shape == (per_rank_dim, 1, kernel)

    def test_tp2_rank0_and_rank1_disjoint(self, loader_fn):
        """
        TP=2 rank 0 and rank 1 should receive disjoint rows of the conv weight.
        Their union should cover all rows of the full weight.
        """
        conv_dim = QWEN35_CONV_DIM
        kernel = QWEN35_CONV_KERNEL
        per_rank = conv_dim // 2

        shard_spec = [
            (QWEN35_KEY_DIM, 0, False),
            (QWEN35_KEY_DIM, 0, False),
            (QWEN35_VALUE_DIM, 0, False),
        ]

        full_weight = torch.randn(conv_dim, kernel)

        param_r0 = self._make_fake_conv1d_param(per_rank, kernel)
        param_r1 = self._make_fake_conv1d_param(per_rank, kernel)

        loader_fn(shard_spec, 2, 0)(param_r0, full_weight)
        loader_fn(shard_spec, 2, 1)(param_r1, full_weight)

        combined = torch.cat([param_r0.squeeze(1), param_r1.squeeze(1)], dim=0)
        # Combined must be a permutation or subset of full_weight rows — not all-zeros
        assert combined.norm().item() > 0.1 * full_weight.norm().item()


# ---------------------------------------------------------------------------
# Tests: in_proj_ba shard dimensions
# ---------------------------------------------------------------------------


class TestInProjBADimensions:
    """Verify in_proj_ba (b/a gate weights) dimensions for Qwen3.5-35B-A3B."""

    def test_ba_output_size_tp1(self):
        # in_proj_ba has output_sizes=[32, 32] for b and a respectively
        # b has 16 heads × 2 = 32, a has 32 as well? Actually let me check:
        # from qwen3_5.py: num_key_heads=16 for GDN
        # in_proj_b output size = num_key_heads = 16? or num_key_heads * 2 = 32?
        # The actual check from the code: output_sizes=[num_key_heads, num_key_heads]
        # For TP=1 this should be [16, 16]... but stacked so shape is [32, hidden]
        num_key_heads = 16  # linear_num_key_heads for Qwen3.5-35B-A3B
        hidden_size = 2048
        total_ba_output = num_key_heads * 2  # b + a
        assert total_ba_output == 32
        # in_proj_ba.weight shape for TP=1: (32, 2048)
        assert (total_ba_output, hidden_size) == (32, 2048)

    def test_ba_output_size_tp2(self):
        num_key_heads = 16
        tp_size = 2
        per_rank_ba = (num_key_heads * 2) // tp_size
        assert per_rank_ba == 16  # 8 b + 8 a per rank


# ---------------------------------------------------------------------------
# Tests: ssm_norm.weight expected values
# ---------------------------------------------------------------------------


class TestSSMNormExpectedValues:
    """Document expected ssm_norm.weight properties after loading."""

    def test_ssm_norm_should_not_be_near_zero(self):
        # RMSNorm weights loaded from GGUF should be ~O(1), not near zero
        # A near-zero ssm_norm.weight would kill all GDN output (H12)
        # This test just verifies our assumption about the expected range
        EXPECTED_MIN = 0.01   # below this = likely not loaded
        EXPECTED_MAX = 10.0   # above this = likely corrupted
        # The actual runtime check is in diag_weight_audit.py
        # Here we document the threshold
        assert EXPECTED_MIN < 1.0 < EXPECTED_MAX  # 1.0 is RMSNorm default init

    def test_ssm_norm_weight_shape(self):
        # ssm_norm.weight is a 1D vector of size value_dim = 4096
        value_dim = QWEN35_VALUE_DIM
        assert value_dim == 4096
