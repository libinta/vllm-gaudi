# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemma4 sliding-window KV cache spec (Option A)."""
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm_gaudi.v1.worker.hpu_model_runner import (
    HPUModelRunner,
    _sliding_group_max_blocks_per_req,
    _sliding_window_kv_enabled,
)


# --------------------------------------------------------------------------
# Task 1: flag gating
# --------------------------------------------------------------------------
def _mk_config(*, interleaved, disable_hybrid, kv_transfer=None,
               use_eagle=False, attention_chunk_size=None):
    """Minimal VllmConfig-shaped stub for the gating predicate."""
    hf_text = SimpleNamespace(sliding_window=1024 if interleaved else None,
                              attention_chunk_size=attention_chunk_size)
    model_config = SimpleNamespace(hf_text_config=hf_text,
                                   attention_chunk_size=attention_chunk_size)
    scheduler_config = SimpleNamespace(
        disable_hybrid_kv_cache_manager=disable_hybrid)
    speculative_config = (SimpleNamespace(use_eagle=lambda: True)
                          if use_eagle else None)
    return SimpleNamespace(model_config=model_config,
                           scheduler_config=scheduler_config,
                           speculative_config=speculative_config,
                           kv_transfer_config=kv_transfer)


def test_flag_off_by_default_disables(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_SLIDING_WINDOW_KV", raising=False)
    cfg = _mk_config(interleaved=True, disable_hybrid=False)
    with patch("vllm_gaudi.v1.worker.hpu_model_runner.is_interleaved",
               return_value=True):
        assert _sliding_window_kv_enabled(cfg) is False


def test_flag_on_enables_for_interleaved_swa(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_SLIDING_WINDOW_KV", "1")
    cfg = _mk_config(interleaved=True, disable_hybrid=False)
    with patch("vllm_gaudi.v1.worker.hpu_model_runner.is_interleaved",
               return_value=True):
        assert _sliding_window_kv_enabled(cfg) is True


def test_flag_on_but_hybrid_disabled_stays_off(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_SLIDING_WINDOW_KV", "1")
    cfg = _mk_config(interleaved=True, disable_hybrid=True)
    with patch("vllm_gaudi.v1.worker.hpu_model_runner.is_interleaved",
               return_value=True):
        assert _sliding_window_kv_enabled(cfg) is False


def test_flag_on_but_not_interleaved_stays_off(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_SLIDING_WINDOW_KV", "1")
    cfg = _mk_config(interleaved=False, disable_hybrid=False)
    with patch("vllm_gaudi.v1.worker.hpu_model_runner.is_interleaved",
               return_value=False):
        assert _sliding_window_kv_enabled(cfg) is False


# --------------------------------------------------------------------------
# Task 2: per-layer spec emission
# --------------------------------------------------------------------------
from vllm.model_executor.layers.attention.attention import Attention  # noqa: E402
from vllm.v1.attention.backend import AttentionType  # noqa: E402


def _mk_attn_module(*, sliding_window, head_size, num_kv_heads):
    m = MagicMock(spec=Attention)
    m.attn_type = AttentionType.DECODER
    m.sliding_window = sliding_window
    m.head_size = head_size
    m.num_kv_heads = num_kv_heads
    m.kv_sharing_target_layer_name = None
    return m


def _spec_from_layers(layers, *, use_swa, block_size=128, dtype="bf16"):
    """Drive the spec-emission branch in isolation via a stub runner."""
    runner = MagicMock(spec=HPUModelRunner)
    runner.use_sliding_window_kv = use_swa
    runner.kv_cache_dtype = dtype
    runner.shared_kv_cache_layers = {}
    runner.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=layers),
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype="auto"))
    return HPUModelRunner.get_kv_cache_spec(runner)


def test_sliding_layer_emits_sliding_spec_when_enabled():
    layers = {
        "l0": _mk_attn_module(sliding_window=1024, head_size=256, num_kv_heads=8),
        "l5": _mk_attn_module(sliding_window=None, head_size=512, num_kv_heads=2),
    }
    spec = _spec_from_layers(layers, use_swa=True)
    assert isinstance(spec["l0"], SlidingWindowSpec)
    assert spec["l0"].sliding_window == 1024
    assert isinstance(spec["l5"], FullAttentionSpec)


def test_all_full_when_disabled():
    layers = {
        "l0": _mk_attn_module(sliding_window=1024, head_size=256, num_kv_heads=8),
        "l5": _mk_attn_module(sliding_window=None, head_size=512, num_kv_heads=2),
    }
    spec = _spec_from_layers(layers, use_swa=False)
    assert isinstance(spec["l0"], FullAttentionSpec)
    assert isinstance(spec["l5"], FullAttentionSpec)


# --------------------------------------------------------------------------
# Task 3: full/sliding attention group resolution
# --------------------------------------------------------------------------
def test_group_id_resolution():
    runner = MagicMock(spec=HPUModelRunner)
    runner.use_sliding_window_kv = True
    runner.num_mamba_like_layers = 0
    full = SimpleNamespace(kv_cache_spec=MagicMock(spec=FullAttentionSpec))
    swa = SimpleNamespace(kv_cache_spec=MagicMock(spec=SlidingWindowSpec))
    # Coordinator sorts full first, sliding second.
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[full, swa])
    assert HPUModelRunner._get_full_attention_group_id(runner) == 0
    assert HPUModelRunner._get_sliding_attention_group_id(runner) == 1


def test_sliding_group_id_none_when_disabled():
    runner = MagicMock(spec=HPUModelRunner)
    runner.use_sliding_window_kv = False
    full = SimpleNamespace(kv_cache_spec=MagicMock(spec=FullAttentionSpec))
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[full])
    assert HPUModelRunner._get_sliding_attention_group_id(runner) is None


# --------------------------------------------------------------------------
# Task 5: window-relative prefill mask
# --------------------------------------------------------------------------
def _past_mask(context_len, seq_len, window, K_cols, base):
    """Reference for the fixed past_mask (single request), matching the
    production formula."""
    ctx = torch.tensor([context_len])
    invalid = ctx - window + torch.arange(seq_len) - 1          # [seq_len]
    past_indices = base + torch.arange(K_cols)                  # [K_cols] absolute
    return ((past_indices.unsqueeze(0) > invalid.unsqueeze(-1)) &
            (past_indices.unsqueeze(0) < ctx.unsqueeze(-1)))    # [seq_len, K_cols]


def test_window_relative_mask_not_all_false_when_context_evicted():
    # Long context, small window: with the base offset the mask selects the
    # in-window tail columns (non-empty). base=0 (the bug) yields all-False.
    context_len = 100_000
    window = 1024
    block_size = 128
    seq_len = 128
    K = math.ceil((window - 1 + seq_len) / block_size) + 1   # kept blocks
    K_cols = K * block_size
    base = max(0, math.ceil(context_len / block_size) * block_size - K_cols)

    pm = _past_mask(context_len, seq_len, window, K_cols, base)
    assert pm.any(), "window-relative past_mask must be non-empty"

    pm_buggy = _past_mask(context_len, seq_len, window, K_cols, base=0)
    assert not pm_buggy.any(), "zero-base reproduces the bug (all-False)"


def test_production_mask_non_empty_with_flag_on(monkeypatch):
    """Drive the real _set_attn_bias_for_sliding_window with Option A on and a
    window-truncated block_list; the produced window_attn_bias must contain
    finite (attended) entries in the context region, not be all -inf."""
    import vllm_gaudi.v1.worker.hpu_model_runner as hm
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUAttentionMetadataProcessor

    window = 1024
    block_size = 128
    seq_len = 128
    context_len = 100_000
    batch_size = 1
    K = math.ceil((window - 1 + seq_len) / block_size) + 1  # kept blocks
    device = torch.device("cpu")
    dtype = torch.float32

    # block_list is the window-truncated set: K blocks * batch, flat.
    block_list = torch.arange(K * batch_size, dtype=torch.long)
    attn_metadata = SimpleNamespace(
        is_prompt=True,
        block_list=block_list,
        context_lens_tensor=torch.tensor([context_len], dtype=torch.int32),
        block_size=block_size)

    proc = MagicMock(spec=HPUAttentionMetadataProcessor)
    proc.use_sliding_window_kv = True
    proc.prefill_use_fusedsdpa = True
    proc.use_window_sdpa = False
    proc.slice_thld = 1 << 30
    proc.slice_size = 0
    proc.block_size = block_size

    captured = {}

    def _fake_replace(obj, typename, **over):
        captured.update(over)
        return obj

    monkeypatch.setattr(hm, "custom_tuple_replace", _fake_replace)

    HPUAttentionMetadataProcessor._set_attn_bias_for_sliding_window(
        proc, attn_metadata, batch_size, seq_len, window, device, dtype)

    bias = captured["window_attn_bias"]
    # Shape: [batch, 1, seq_len, K*block_size + seq_len]
    assert bias.shape[0] == batch_size
    assert bias.shape[2] == seq_len
    # Context region (first K*block_size columns) must have finite entries for
    # the last query row -> the layer attends to in-window cached context.
    ctx_cols = K * block_size
    last_row_ctx = bias[0, 0, -1, :ctx_cols]
    assert torch.isfinite(last_row_ctx).any(), \
        "window-relative mask must attend to some cached context (not all -inf)"
