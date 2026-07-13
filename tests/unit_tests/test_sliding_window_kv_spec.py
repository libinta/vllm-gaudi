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
