# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Unit + accuracy tests for window-aware SlicedFusedSDPA.

Makes the chunked FusedSDPA path sliding-window aware: KV chunks that
fall entirely outside the sliding window are skipped (saving prefill compute),
while the window band itself stays encoded in the attention mask.

The pure-logic tests here run on CPU-in-container (no HPU); the accuracy test
runs on device and compares the window-sliced output against a single-call
reference that also uses the same window mask -- they must match, because
skipped chunks contribute nothing to the online softmax.
"""

import math
import torch
import pytest
from unittest.mock import patch, MagicMock

from vllm_gaudi.extension.utils import (
    ModuleFusedSDPA,
    SlicedFusedSDPA,
    SlicedFusedSDPABase,
)

# Reuse the shared HPU-accuracy helpers from the base slicing test module.
from tests.unit_tests.test_fsdpa_slicing import (
    _hpu_available,
    _make_sliced_bf16,
    _generate_realistic_qkv,
)

requires_hpu = pytest.mark.skipif(not _hpu_available(), reason="HPU device not available")


# ---------------------------------------------------------------------------
# _merge_chunk: tolerate -inf-fill masks (fully-masked rows)
# ---------------------------------------------------------------------------


class TestMergeChunkNegInf:
    """_merge_chunk must not produce NaN when a chunk has a fully-masked row.

    Masks that use -inf fill (e.g. Gemma's sliding-window / causal masks) give a
    fully-masked row a kernel row-max of -inf and a nan chunk output. The naive
    online-softmax rescale exp(-inf - -inf) = nan then poisons the whole merge
    (observed as <pad> garbage end-to-end). The merge must instead treat such
    rows as zero-weight and stay finite.
    """

    def _merge(self, last, chunk):
        # last/chunk are (out, m, linv) tuples of float32 tensors
        return SlicedFusedSDPABase._merge_chunk(*last, *chunk)

    # Kernel returns out=[bs, heads, q, d]; m, linv = [bs, heads, q, 1].

    def test_fully_masked_chunk_row_does_not_nan(self):
        # `last` chunk valid; `chunk` fully masked for the row (m=-inf, nan out
        # as the kernel emits).
        d = 4
        last_out = torch.randn(1, 1, 1, d)
        last_m = torch.zeros(1, 1, 1, 1)  # finite row-max
        last_linv = torch.full((1, 1, 1, 1), 0.5)
        chunk_out = torch.full((1, 1, 1, d), float('nan'))  # kernel nan for masked row
        chunk_m = torch.full((1, 1, 1, 1), float('-inf'))
        chunk_linv = torch.full((1, 1, 1, 1), float('inf'))
        out, m, linv = self._merge((last_out, last_m, last_linv), (chunk_out, chunk_m, chunk_linv))
        assert not torch.isnan(out).any(), "merge produced NaN on fully-masked chunk row"
        assert not torch.isinf(out).any()
        # a fully-masked chunk contributes nothing => merged output equals `last`
        torch.testing.assert_close(out, last_out, rtol=1e-4, atol=1e-4)

    def test_both_chunks_fully_masked_row_stays_finite(self):
        # A query row masked in BOTH chunks (e.g. query-tail padding): discarded
        # downstream, but must be finite so it doesn't poison later ops.
        d = 4
        neg = float('-inf')
        last = (torch.full((1, 1, 1, d), float('nan')), torch.full((1, 1, 1, 1), neg), torch.full(
            (1, 1, 1, 1), float('inf')))
        chunk = (torch.full((1, 1, 1, d), float('nan')), torch.full((1, 1, 1, 1), neg), torch.full(
            (1, 1, 1, 1), float('inf')))
        out, m, linv = self._merge(last, chunk)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()

    def test_valid_rows_unchanged_vs_naive(self):
        # For finite row-maxes the fix must match the original online-softmax math.
        d = 8
        torch.manual_seed(0)
        last_out = torch.randn(1, 2, 3, d)
        last_m = torch.randn(1, 2, 3, 1)
        last_linv = torch.rand(1, 2, 3, 1) + 0.5
        chunk_out = torch.randn(1, 2, 3, d)
        chunk_m = torch.randn(1, 2, 3, 1)
        chunk_linv = torch.rand(1, 2, 3, 1) + 0.5
        out, m, linv = self._merge((last_out, last_m, last_linv), (chunk_out, chunk_m, chunk_linv))
        # reference naive merge
        new_m = torch.maximum(last_m, chunk_m)
        lr = (1.0 / last_linv) * torch.exp(last_m - new_m)
        cr = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
        nl = 1.0 / (lr + cr)
        ref = (lr * nl) * last_out + (cr * nl) * chunk_out
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Pure-logic tests for the out-of-window skip predicate
# ---------------------------------------------------------------------------


class TestChunkFullyMasked:
    """_chunk_fully_masked(mask_slice): a chunk whose mask slice is entirely at
    or below the mask-off threshold contributes nothing and can be skipped.

    This is mask-driven (not position-driven) so it stays exact even when the
    cached context is padded (padded context shifts the chunk loop's prefix_len
    relative to the window origin encoded in the mask).
    """

    pred = staticmethod(SlicedFusedSDPABase._chunk_fully_masked)

    def test_all_off_is_skippable(self):
        m = torch.full((1, 1, 128, 256), -3e38)
        assert self.pred(m) is True

    def test_all_off_neg_inf_is_skippable(self):
        m = torch.full((1, 1, 128, 256), float('-inf'))
        assert self.pred(m) is True

    def test_any_in_window_entry_is_not_skippable(self):
        m = torch.full((1, 1, 128, 256), -3e38)
        m[0, 0, 0, 0] = 0.0  # one in-window position
        assert self.pred(m) is False

    def test_all_zero_is_not_skippable(self):
        m = torch.zeros((1, 1, 128, 256))
        assert self.pred(m) is False

    def test_finite_negative_bias_is_not_skippable(self):
        # a real (small) negative bias must NOT be treated as masked-off
        m = torch.full((1, 1, 128, 256), -5.0)
        assert self.pred(m) is False


# ---------------------------------------------------------------------------
# Dispatch tests: window slicing is gated behind the opt-in flag
# ---------------------------------------------------------------------------


class TestWindowSlicingDispatch:
    """ModuleFusedSDPA.forward routes sliding-window prefill to the sliced
    module only when the opt-in flag (_enable_window_slicing) is set and the
    window is causal (right == 0)."""

    def _make_module(self, enable_window_slicing, slice_thld=4096):
        mock_kernel = MagicMock()
        mock_kernel.apply.return_value = torch.zeros(1, 4, 2048, 64)

        with patch('vllm_gaudi.extension.utils.SlicedFusedSDPABase.__init__', return_value=None):
            module = ModuleFusedSDPA.__new__(ModuleFusedSDPA)
            torch.nn.Module.__init__(module)
            module._hpu_kernel_fsdpa = mock_kernel
            mock_sliced = MagicMock(return_value=torch.zeros(1, 4, 2048, 64))
            mock_sliced.enable_slicing = True
            mock_sliced.slice_thld = slice_thld
            mock_sliced._enable_window_slicing = enable_window_slicing
            module._sliced_module = mock_sliced
        return module

    def _args(self):
        q = torch.randn(1, 4, 2048, 64)
        k = torch.randn(1, 4, 8192, 64)
        v = torch.randn(1, 4, 8192, 64)
        mask = torch.zeros(1, 1, 2048, 8192)
        return q, k, v, mask

    def test_window_routed_to_sliced_when_flag_on(self):
        module = self._make_module(enable_window_slicing=True)
        q, k, v, mask = self._args()
        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right',
                       window_size=(1024, 0))
        module._sliced_module.assert_called_once()
        # window_size must be threaded into the sliced module
        _, kwargs = module._sliced_module.call_args
        assert kwargs.get('window_size') == (1024, 0)

    def test_window_not_routed_when_flag_off(self):
        module = self._make_module(enable_window_slicing=False)
        q, k, v, mask = self._args()
        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right',
                       window_size=(1024, 0))
        module._sliced_module.assert_not_called()
        module._hpu_kernel_fsdpa.apply.assert_called_once()

    def test_non_causal_window_not_routed_even_with_flag(self):
        # right != 0 (bidirectional window) is not supported by the skip logic
        module = self._make_module(enable_window_slicing=True)
        q, k, v, mask = self._args()
        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right',
                       window_size=(1024, 512))
        module._sliced_module.assert_not_called()

    def test_non_window_still_sliced_with_flag_on(self):
        # regression: enabling window slicing must not change non-window routing
        module = self._make_module(enable_window_slicing=True)
        q, k, v, mask = self._args()
        module.forward(q, k, v, mask, 0.0, True, None, 'fast', True, None, padding_side='right')
        module._sliced_module.assert_called_once()
        _, kwargs = module._sliced_module.call_args
        assert kwargs.get('window_size') is None


# ---------------------------------------------------------------------------
# Setup flag test
# ---------------------------------------------------------------------------


class TestWindowSlicingSetupFlag:

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_window_slicing_flag_defaults_off(self, mock_get_instance, mock_get_config, mock_is_lazy):
        from tests.unit_tests.test_fsdpa_slicing import _make_config, _MockBucketingManager
        mock_get_config.return_value = _make_config(enable_fsdpa_window_slicing=False)
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        assert base._setup_slicing() is True
        assert base._enable_window_slicing is False

    @patch('habana_frameworks.torch.utils.internal.is_lazy', return_value=False)
    @patch('vllm_gaudi.extension.utils.get_config')
    @patch('vllm_gaudi.extension.bucketing.common.HPUBucketingManager.get_instance')
    def test_window_slicing_flag_enables(self, mock_get_instance, mock_get_config, mock_is_lazy):
        from tests.unit_tests.test_fsdpa_slicing import _make_config, _MockBucketingManager
        mock_get_config.return_value = _make_config(enable_fsdpa_window_slicing=True)
        mock_get_instance.return_value = _MockBucketingManager(max_num_batched_tokens=8192, block_size=128)
        base = SlicedFusedSDPABase.__new__(SlicedFusedSDPABase)
        assert base._setup_slicing() is True
        assert base._enable_window_slicing is True


# ---------------------------------------------------------------------------
# Device accuracy: window-sliced output must match a windowed single-call ref
# ---------------------------------------------------------------------------


def _build_windowed_causal_mask(seq_len_valid, seq_len_pad, ctx_len_valid, ctx_len_pad, window_left, device,
                                dtype=torch.bfloat16):
    """Full [1,1,q_pad,ctx_pad+q_pad] mask built exactly like the production
    ``_set_attn_bias_for_sliding_window`` block-list branch: the past-context
    region and the current-query region are constructed *separately*.

    - past region (width ctx_len_pad): key j kept iff
      ``context_len - window + i - 1 < j < context_len`` (window lower edge +
      valid-context upper edge; masks padded context).
    - current region (width seq_len_pad): plain tril/triu band -- current key
      qq kept iff ``i - window + 1 <= qq <= i`` (causal + window), *independent*
      of context padding. This is the detail a unified-absolute-position mask
      gets wrong.

    ``seq_len_valid`` is accepted for signature compatibility; the production
    builder does not mask query-tail padding here (see its commented-out
    len_mask), so we don't either.
    """
    off_value = -3e38  # finite (avoids nan in exp during chunk rescaling)
    i = torch.arange(seq_len_pad, device=device).view(-1, 1)

    # past-context region
    jc = torch.arange(ctx_len_pad, device=device).view(1, -1)
    invalid = ctx_len_valid - window_left + i - 1
    past_keep = (jc > invalid) & (jc < ctx_len_valid)

    # current-query region: tril(diag=0) & triu(diag=-window+1)
    qq = torch.arange(seq_len_pad, device=device).view(1, -1)
    cur_keep = (qq <= i) & (qq >= i - window_left + 1)

    keep = torch.cat([past_keep, cur_keep], dim=-1)
    bias = torch.zeros((seq_len_pad, ctx_len_pad + seq_len_pad), dtype=dtype,
                       device=device).masked_fill(~keep, off_value)
    return bias.view(1, 1, seq_len_pad, ctx_len_pad + seq_len_pad)


@requires_hpu
class TestWindowSlicingAccuracy:
    """Correctness of the window skip optimization.

    The claim is *narrow and exact*: skipping KV chunks that fall
    entirely outside the sliding window changes nothing, because those chunks
    are exactly the ones the window mask already sends to -inf (they add zero
    weight to the online softmax). We therefore compare the chunked path WITH
    skipping against the same chunked path WITHOUT skipping, both fed the same
    full window mask. This isolates the optimization from the (pre-existing)
    numerical difference between the chunked kernel and a single-call kernel,
    which is unrelated to the window skip.

    A separate looser check confirms the chunked window output still tracks a
    single-call reference (same tolerance regime as the base slicing tests).
    """

    @staticmethod
    def _run_reference(q, k, v, attn_mask):
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        with torch.inference_mode():
            out = FusedSDPA.apply(q, k, v, attn_mask, 0.0, False, None, 'fast', True, None, 'right')
        torch.hpu.synchronize()
        return out

    @staticmethod
    def _run_chunked(q, k, v, attn_mask, chunk_size, q_pad, ctx_pad, window_size):
        module = _make_sliced_bf16(chunk_size, math.ceil(q_pad / chunk_size), math.ceil(ctx_pad / chunk_size), False)
        module = module.to('hpu')
        with torch.inference_mode():
            out = module(q, k, v, attn_mask, 0.0, True, None, 'fast', window_size=window_size)
        torch.hpu.synchronize()
        return out

    @staticmethod
    def _run_chunked_no_skip(q, k, v, attn_mask, chunk_size, q_pad, ctx_pad, window_size):
        # Force the skip predicate off so every chunk is processed (still with
        # the window mask). This is the reference the skip must not perturb.
        with patch('vllm_gaudi.extension.utils.SlicedFusedSDPABase._chunk_fully_masked',
                   classmethod(lambda cls, *a, **k: False)):
            return TestWindowSlicingAccuracy._run_chunked(q, k, v, attn_mask, chunk_size, q_pad, ctx_pad, window_size)

    @pytest.mark.parametrize("q_len,ctx_len,chunk_size,window", [
        (2048, 8192, 2048, 1024),
        (2048, 8192, 2048, 4096),
        (1024, 8192, 4096, 1024),
        (1024, 8192, 2048, 1024),
    ])
    def test_skip_does_not_change_chunked_output(self, q_len, ctx_len, chunk_size, window):
        """The window skip is exact: with-skip == no-skip (same window mask)."""
        bs, heads, kv_heads, head_dim, pad = 1, 8, 2, 128, 128
        q_len_pad = q_len + pad
        ctx_len_pad = ctx_len + pad
        kv_len_pad = q_len_pad + ctx_len_pad

        q, k, v = _generate_realistic_qkv(bs, heads, kv_heads, head_dim, q_len_pad, kv_len_pad, device='hpu')
        mask = _build_windowed_causal_mask(q_len, q_len_pad, ctx_len, ctx_len_pad, window, device='hpu')

        with_skip = self._run_chunked(q, k, v, mask, chunk_size, pad, pad, window_size=(window, 0))
        no_skip = self._run_chunked_no_skip(q, k, v, mask, chunk_size, pad, pad, window_size=(window, 0))

        a = with_skip[:, :, :q_len, :].flatten().float()
        b = no_skip[:, :, :q_len, :].flatten().float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        assert cos > 0.9999, f"window skip changed the output (cos={cos}); it must be exact"

    @staticmethod
    def _run_chunked_band(q, k, v, chunk_size, q_pad, ctx_pad, window_size, context_len):
        """Memory-win path: no full mask, band masks built on the fly from
        context_len."""
        module = _make_sliced_bf16(chunk_size, math.ceil(q_pad / chunk_size), math.ceil(ctx_pad / chunk_size), False)
        module = module.to('hpu')
        with torch.inference_mode():
            out = module(q, k, v, None, 0.0, True, None, 'fast', window_size=window_size, context_len=context_len)
        torch.hpu.synchronize()
        return out

    @pytest.mark.parametrize("q_len,ctx_len,chunk_size,window", [
        (2048, 8192, 2048, 1024),
        (2048, 8192, 2048, 4096),
        (1024, 8192, 4096, 1024),
        (1024, 8192, 2048, 1024),
    ])
    def test_band_mask_matches_full_mask(self, q_len, ctx_len, chunk_size, window):
        """Memory win: building band masks on the fly (attn_mask=None,
        context_len given) matches slicing the materialized full window mask."""
        bs, heads, kv_heads, head_dim, pad = 1, 8, 2, 128, 128
        q_len_pad = q_len + pad
        ctx_len_pad = ctx_len + pad
        kv_len_pad = q_len_pad + ctx_len_pad

        q, k, v = _generate_realistic_qkv(bs, heads, kv_heads, head_dim, q_len_pad, kv_len_pad, device='hpu')
        # full-mask reference uses ctx_len_pad as the current-query offset, so
        # prefix_len == ctx_len_pad and the valid context length is ctx_len.
        mask = _build_windowed_causal_mask(q_len, q_len_pad, ctx_len, ctx_len_pad, window, device='hpu')

        full = self._run_chunked(q, k, v, mask, chunk_size, pad, pad, window_size=(window, 0))
        band = self._run_chunked_band(q, k, v, chunk_size, pad, pad, window_size=(window, 0), context_len=ctx_len)

        a = full[:, :, :q_len, :].flatten().float()
        b = band[:, :, :q_len, :].flatten().float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        assert cos > 0.9999, f"band-mask path diverged from full-mask path (cos={cos})"

    @pytest.mark.parametrize("q_len,ctx_len,chunk_size,window", [
        (2048, 8192, 2048, 1024),
    ])
    def test_window_chunked_tracks_single_call(self, q_len, ctx_len, chunk_size, window):
        """Sanity: chunked window output stays close to a single windowed call
        for the multi-q-chunk regime the production path actually runs."""
        bs, heads, kv_heads, head_dim, pad = 1, 8, 2, 128, 128
        q_len_pad = q_len + pad
        ctx_len_pad = ctx_len + pad
        kv_len_pad = q_len_pad + ctx_len_pad

        q, k, v = _generate_realistic_qkv(bs, heads, kv_heads, head_dim, q_len_pad, kv_len_pad, device='hpu')
        mask = _build_windowed_causal_mask(q_len, q_len_pad, ctx_len, ctx_len_pad, window, device='hpu')

        ref = self._run_reference(q, k, v, mask)
        sliced = self._run_chunked(q, k, v, mask, chunk_size, pad, pad, window_size=(window, 0))

        a = ref[:, :, :q_len, :].flatten().float()
        b = sliced[:, :, :q_len, :].flatten().float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        assert cos > 0.999, f"window-sliced cos sim too low: {cos}"
