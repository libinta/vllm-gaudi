###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os
import math
from functools import lru_cache, wraps
from typing import Optional, Any

import habana_frameworks.torch as htorch
import torch
import itertools
from vllm_gaudi.extension.logger import logger

from vllm_gaudi.extension.runtime import get_config


@lru_cache(maxsize=None)
def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y, **kwargs):
        return torch.matmul(x, y, **kwargs)


class B2BMatmul(Matmul):
    """Specialized alias for batch2block and block2batch matmul operations.
    
    This class remains functionally identical to ``Matmul`` but is used to
    semantically mark B2B-related matmuls. This enables the system to apply the
    fix that uses the B2B output measurements as the input measurements during
    calibration, avoiding corrupted scales from the KV‑cache.
    """

    def __init__(self):
        super().__init__()


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


def get_kv_fetch_extra_args(**kwargs):
    if not get_config().per_token_kv_scaling_support:
        kwargs.pop('scales', None)
    return kwargs


class VLLMKVCache(torch.nn.Module):

    def __init__(self, is_v_cache: bool = False):
        super().__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        # is_v_cache is used in INC FP8 dynamic quantization to identify V cache
        self.is_v_cache = is_v_cache

    def forward(self, input, cache, slot_mapping, scales=None, block_size=None, is_prompt=False, **kwargs):
        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            cache.index_copy_(0, slot_mapping, input)
        return cache

    def fetch_from_cache(self, cache, blocks, scales=None, **kwargs):
        # Contiguous PA's fast path (cache[:n]) is only valid when `blocks` is the
        # contiguous-identity layout the decode path builds (blocks[i] == i). The
        # prefill-context path passes raw, scattered physical block ids, for which
        # cache[:n] would return the wrong rows (GAUDISW-248985). The prefill path
        # sets `_fetch_by_id` on this module so we gather by id there, while decode
        # keeps the zero-copy slice. We use an instance flag rather than a kwarg
        # because INC's PatchedVLLMKVCache wraps this method with a fixed signature.
        if self.use_contiguous_pa and not getattr(self, "_fetch_by_id", False):
            return cache[:blocks.size(0)]
        else:
            return cache.index_select(0, blocks)


class VLLMFP8KVCache(VLLMKVCache):

    def __init__(self, input_scale=1.0):
        super().__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.input_scale = input_scale
        self.output_scale = 1.0 / self.input_scale

    def quant_input(self, input):
        return torch.ops.hpu.cast_to_fp8_v2(input, self.input_scale, False, False, torch.float8_e4m3fn)[0]

    def dequant_output(self, output):
        return torch.ops.hpu.cast_from_fp8(output, self.output_scale, torch.bfloat16)

    def forward(self, input, *args, **kwargs):
        qinput = self.quant_input(input)
        return super().forward(qinput, *args, **kwargs)

    def fetch_from_cache(self, quant_cache, blocks, permutations=None, **kwargs):
        if permutations:
            output_cache = super().fetch_from_cache(quant_cache, blocks, permutations)
            for i in range(len(output_cache)):
                output_cache[i] = self.dequant_output(output_cache[i])
            return output_cache
        output_cache = super().fetch_from_cache(quant_cache, blocks)
        return self.dequant_output(output_cache)


class FP8Matmul(torch.nn.Module):

    def __init__(
        self,
        scale_input=1.0,
        scale_other=1.0,
    ):
        super().__init__()
        self.scale_input = scale_input
        self.scale_other = scale_other

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, torch.float8_e4m3fn)[0]

    def matmul_fp8(self, x, other, out_dtype, scale_input_inv=None, scale_other_inv=None):
        return torch.ops.hpu.fp8_gemm_v2(
            A=x,
            trans_A=False,
            B=other,
            trans_B=False,
            D=None,
            out_dtype=out_dtype,
            A_scale_inv=scale_input_inv,
            B_scale_inv=scale_other_inv,
            bias=None,
            accumulate=False,
        )

    def forward(self, input, other, **kwargs):
        qinput = self.quant_input(input, self.scale_input)
        qother = self.quant_input(other, self.scale_other)
        output = self.matmul_fp8(
            qinput,
            qother,
            out_dtype=torch.bfloat16,
            scale_input_inv=1.0 / self.scale_input,
            scale_other_inv=1.0 / self.scale_other,
        )
        return output


def window_slicing_will_skip_chunks(prefix_len: int, chunk_size: int, window_left: int) -> bool:
    """Return True when window-aware slicing will skip at least one KV chunk.

    Window-aware SlicedFusedSDPA only provides a net benefit when at least one
    KV chunk falls entirely outside the sliding window and can be skipped.
    The HEAD chunk covers [0, chunk_size); it is skippable iff its end falls
    below the window lower bound for the earliest query token:
        chunk_size <= prefix_len - window_left
        → prefix_len >= chunk_size + window_left

    This predicate must be evaluated identically in:
      1. ModuleFusedSDPA.forward() (utils.py) — to decide whether to route
         the attention call through the sliced path.
      2. _set_attn_bias_for_sliding_window() (hpu_model_runner.py) — to
         decide whether to skip materialising the full window mask (memory win).
    Keeping both callers in sync avoids either missing-mask or wasted-mask bugs.
    """
    return prefix_len >= chunk_size + window_left


class SlicedFusedSDPABase(torch.nn.Module):
    """Base class for sliced FusedSDPA modules.

    Encapsulates the common slicing initialization (chunk size, padded chunk
    counts, graph-break setup) shared by :class:`SlicedFusedSDPA` and
    :class:`SlicedFP8FusedSDPA`.
    """

    def __init__(self):
        super().__init__()
        self.enable_slicing = self._setup_slicing()

    @staticmethod
    def _max_bucket_gap(bucket_values: list[int]) -> int:
        """Largest gap between consecutive sorted bucket values.

        This bounds the worst-case padding (in the same units as
        ``bucket_values``) that any request routed through this bucket set can
        receive: a request whose true size falls just above bucket[i] pads up
        to bucket[i+1], i.e. by at most ``bucket[i+1] - bucket[i] - 1``.
        """
        if len(bucket_values) < 2:
            return 0
        vals = sorted(set(bucket_values))
        return max(b - a - 1 for a, b in zip(vals, vals[1:]))

    def _setup_slicing(self) -> bool:
        enable_slicing = get_config().enable_fsdpa_slicing
        if not enable_slicing:
            return False

        if get_config().merged_prefill:
            logger().warning_once(
                'FusedSDPA slicing is not compatible with merged prefill, slicing in FusedSDPA will be disabled.')
            return False

        if not get_config().use_bucketing:
            logger().warning_once(
                'FusedSDPA slicing requires bucketing to be enabled, slicing in FusedSDPA will be disabled.')
            return False

        from vllm_gaudi.extension.bucketing.common import get_bucketing_manager
        bucketing_manager = get_bucketing_manager()
        assert bucketing_manager is not None and bucketing_manager.initialized, 'Bucketing manager should be instantiated and initialized to enable FusedSDPA slicing.'

        from vllm_gaudi.extension.bucketing.padding_aware import PaddingAwareBucketingStrategy
        strategy = bucketing_manager.get_bucketing_strategy()
        is_pad_strategy = isinstance(strategy, PaddingAwareBucketingStrategy)

        max_num_batched_tokens = bucketing_manager.max_num_batched_tokens
        block_size = bucketing_manager.block_size
        slice_thld_default = min(max_num_batched_tokens, 8192)
        slice_thld = int(os.getenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", str(slice_thld_default)))
        assert slice_thld > block_size, 'Invalid FusedSDPA slice sequence length threshold, the threshold should be greater than the block size.'
        assert slice_thld >= 1024, 'The FusedSDPA slice sequence length threshold should be greater than or equal to 1024 to ensure the chunk sizes are valid for the attention kernel.'
        if slice_thld < slice_thld_default:
            logger().warning_once(
                f'The FusedSDPA slice sequence length threshold {slice_thld} is less than the default {slice_thld_default} which is not recommended.'
            )

        # defaults to half of the threshold and round up by 1024
        chunk_size_default = math.ceil(slice_thld // 2 / 1024) * 1024
        chunk_size = int(os.getenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", str(chunk_size_default)))
        if chunk_size % 1024 != 0:
            chunk_size = math.ceil(chunk_size / 1024) * 1024
            logger().warning_once('Rounded up the chunk size for FusedSDPA slicing to the next multiple of 1024.')
        assert chunk_size > block_size and chunk_size <= slice_thld, 'Invalid FusedSDPA slice chunk size, the chunk size should be between the block size and the slice sequence length threshold.'

        if get_config().enable_fsdpa_window_slicing:
            # Non-window (full-attention) layers report window_size=None to
            # ModuleFusedSDPA.forward and so never take the window branch;
            # they fall into the plain base-sliced branch instead, gated only
            # by kv_len >= slice_thld. That base-sliced path is broken for
            # models with heterogeneous per-layer-type attention (e.g.
            # gemma-4: corrupted/garbled output on every request that crosses
            # slice_thld, confirmed via real-prompt replay). Forcing slice_thld
            # up to max_model_len (AFTER deriving chunk_size above, so the
            # window-slicing chunk size stays small) means no real request's
            # full-attention layers can ever satisfy kv_len >= slice_thld, so
            # they always fall through to plain (unsliced) FusedSDPA -- while
            # sliding-window layers are unaffected, since their window branch
            # is checked first and gated by chunk_size + window_left, not
            # slice_thld.
            max_model_len = bucketing_manager.max_model_len
            if slice_thld < max_model_len:
                logger().warning_once(
                    f'FusedSDPA window slicing is enabled: raising the slice sequence length threshold from '
                    f'{slice_thld} to max_model_len ({max_model_len}) so non-window (full-attention) layers never '
                    f'take the base sliced path, which is broken for models with heterogeneous per-layer-type '
                    f'attention (e.g. gemma-4). Sliding-window layers and chunk_size are unaffected.')
                slice_thld = max_model_len

        self.slice_thld = slice_thld
        self.chunk_size = chunk_size

        if is_pad_strategy:
            # Padding-aware bucketing exposes an explicit, tunable padding
            # bound (PAD_MAX) -- use it directly, unchanged from before.
            # should align with the default value in PaddingAwareBucketingStrategy
            max_query_pad_default = math.ceil(max_num_batched_tokens / 4)
            max_query_pad = int(os.getenv("VLLM_PROMPT_QUERY_BUCKET_PAD_MAX", str(max_query_pad_default)))
            assert max_query_pad >= block_size, 'Invalid max query padding, the max query padding should be greater than or equal to the block size.'
            self.num_padded_query_chunks = math.ceil(max_query_pad / self.chunk_size)

            # should align with the default value in PaddingAwareBucketingStrategy
            max_ctx_pad_default = math.ceil(max_num_batched_tokens / block_size)
            max_ctx_pad = int(os.getenv("VLLM_PROMPT_CTX_BUCKET_PAD_MAX", str(max_ctx_pad_default)))
            self.num_padded_ctx_chunks = math.ceil(max_ctx_pad * block_size / self.chunk_size)
        else:
            # Any other strategy (exp, lin, ...) has no explicit padding
            # bound, but it still generates a concrete bucket list -- derive
            # the same bound from that list's own worst-case gap. A LARGER
            # num_padded_*_chunks is always safe here (more chunks get an
            # explicit mask than strictly necessary; only an UNDERestimate
            # could skip masking an actually-padded chunk and corrupt output),
            # so this is a strict correctness superset of the pad-strategy
            # path, just computed from a different (implicit) source.
            query_bs_cfg, query_query_cfg, query_ctx_cfg = strategy.get_prompt_cfgs(
                max_num_prefill_seqs=bucketing_manager.max_num_prefill_seqs,
                block_size=block_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=bucketing_manager.max_model_len)
            query_range = strategy.get_range(query_query_cfg)
            ctx_range = strategy.get_range(query_ctx_cfg)

            max_query_pad = self._max_bucket_gap(query_range)
            self.num_padded_query_chunks = math.ceil(max_query_pad / self.chunk_size) if max_query_pad else 0

            max_ctx_pad_blocks = self._max_bucket_gap(ctx_range)
            self.num_padded_ctx_chunks = math.ceil(
                max_ctx_pad_blocks * block_size / self.chunk_size) if max_ctx_pad_blocks else 0

            logger().warning_once(
                f'FusedSDPA slicing: bucketing strategy is not padding-aware; deriving the padded-chunk '
                f'bound from the {get_config().bucketing_strategy} strategy\'s own bucket gaps '
                f'(num_padded_query_chunks={self.num_padded_query_chunks}, '
                f'num_padded_ctx_chunks={self.num_padded_ctx_chunks}). This is safe (never under-masks) '
                f'but may mask more chunks than the padding-aware path would for the same workload.')

        import habana_frameworks.torch as ht
        is_lazy = ht.utils.internal.is_lazy()
        self._with_graph_breaks = os.getenv("VLLM_HPU_FSDPA_SLICE_WITH_GRAPH_BREAKS",
                                            str(is_lazy)).strip().lower() in ['true', 't', '1', 'yes', 'y', 'on']
        if self._with_graph_breaks and not is_lazy:
            logger().warning_once('FusedSDPA slicing graph breaks are only supported in lazy mode. '
                                  'Disabling graph breaks for eager/compile mode to avoid Synapse compiler failures.')
            self._with_graph_breaks = False
        if self._with_graph_breaks:
            self._break_graph = ht.core.mark_step

        # Opt-in extension of slicing to sliding-window layers. When
        # enabled, ModuleFusedSDPA routes prefix-prefill of sliding-window
        # attention through the chunked path, which skips KV chunks that fall
        # entirely outside the window. Off by default so slicing behaviour for
        # non-window layers is unchanged.
        self._enable_window_slicing = get_config().enable_fsdpa_window_slicing

        msg = (f"FusedSDPA slicing is enabled with sequence length threshold {slice_thld}, "
               f"chunk size {self.chunk_size}, num padded query chunks {self.num_padded_query_chunks}, "
               f"num padded ctx chunks {self.num_padded_ctx_chunks}, with graph breaks {self._with_graph_breaks}, "
               f"window slicing {self._enable_window_slicing}.")
        logger().debug_once(msg)

        return True

    def maybe_break_graph(self):
        if self._with_graph_breaks:
            self._break_graph()

    @staticmethod
    def _merge_chunk(last_out, last_m, last_linv, chunk_out, chunk_m, chunk_linv):
        """Online softmax rescaling merge of two attention chunks.

        Tolerates fully-masked rows (row-max ``m == -inf``): a query whose valid
        keys lie entirely in a different chunk yields ``m = -inf`` from the
        kernel, and the naive rescale ``exp(m - new_m)`` becomes
        ``exp(-inf - -inf) = exp(nan) = nan``. This happens whenever the
        attention mask uses ``-inf`` fill (e.g. Gemma's sliding-window mask)
        rather than a large finite value. We clamp those rescale factors to 0 so
        a fully-masked chunk contributes nothing, keeping the merge finite.
        """
        if last_out is None or last_m is None or last_linv is None:
            return chunk_out, chunk_m, chunk_linv
        new_m = torch.maximum(last_m, chunk_m)
        # Fully-masked rows (row-max -inf) need special handling: the kernel
        # returns nan for the chunk output of such a row, and the naive rescale
        # exp(-inf - -inf) = exp(nan) = nan. This happens whenever the attention
        # mask uses -inf fill (e.g. Gemma's sliding-window mask) rather than a
        # large finite value. A fully-masked row must contribute zero weight, so
        # we compute the merge with finite substitutes and select the correct
        # branch per-row with torch.where (multiplying nan by 0 would not work).
        last_masked = (last_m == float('-inf'))
        chunk_masked = (chunk_m == float('-inf'))
        last_scale = torch.where(last_masked, torch.zeros_like(new_m), torch.exp(last_m - new_m))
        chunk_scale = torch.where(chunk_masked, torch.zeros_like(new_m), torch.exp(chunk_m - new_m))
        last_linv_rescaled = (1.0 / last_linv) * last_scale
        chunk_linv_rescaled = (1.0 / chunk_linv) * chunk_scale
        denom = last_linv_rescaled + chunk_linv_rescaled
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        new_linv = 1.0 / safe_denom
        # Zero out nan chunk/last outputs from fully-masked rows before weighting
        # so 0*nan does not leak; masked rows then produce 0 (discarded downstream).
        last_contrib = torch.where(last_masked, torch.zeros_like(last_out),
                                   (last_linv_rescaled * new_linv) * last_out)
        chunk_contrib = torch.where(chunk_masked, torch.zeros_like(chunk_out),
                                    (chunk_linv_rescaled * new_linv) * chunk_out)
        new_out = last_contrib + chunk_contrib
        return new_out, new_m, new_linv

    # Threshold below which a mask entry is treated as "fully masked" (the
    # sliding-window / padding fill uses -3e38 or -inf; any real bias is far
    # above this). Used to decide when a whole KV chunk contributes nothing.
    _MASK_OFF_THRESHOLD = -1e30

    @classmethod
    @torch.compiler.disable
    def _chunk_fully_masked(cls, mask_slice):
        """Return True if every entry of ``mask_slice`` is masked off, so the
        chunk contributes nothing to the online softmax and can be skipped.

        Mask-driven (not position-driven) on purpose: the sliding-window band
        is already encoded in ``attn_mask`` relative to the *valid* context
        length, whereas the chunk loop only knows the *padded* ``prefix_len``.
        Deciding skippability from the mask is therefore exact regardless of
        context padding, where a position-based test would be too aggressive
        and could drop in-window keys.

        ``@torch.compiler.disable``: this is a data-dependent device-tensor
        read (``bool(tensor)``). Under ``torch.compile`` Dynamo cannot trace
        through it and graph-breaks here regardless; without this decorator
        it re-attempts (and re-aborts) tracing into this call on every distinct
        mask shape, which is the dominant recompilation cost under window-aware
        slicing (see the many more distinct "not warmed-up" bucket shapes logged
        with it enabled vs. disabled). Marking it disabled short-circuits
        that analysis to a single cheap, stable break point.
        """
        return bool((mask_slice <= cls._MASK_OFF_THRESHOLD).all())

    @staticmethod
    def _context_chunk_fully_masked_analytic(q_start, q_end, kv_start, kv_end, prefix_len, context_len, window_left):
        """Host-integer equivalent of ``_chunk_fully_masked`` for a context-loop
        chunk (``kv_end <= prefix_len``) in the band-mask (memory-win) path.

        Bit-exact vs. materializing ``_window_band_mask``'s ``keep`` grid and
        checking ``not keep.any()`` (verified by brute-force + exhaustive sweep
        over chunk sizes, prefix/context lengths, and window sizes including
        window_left <= 0). Avoids the device tensor + ``bool()`` sync entirely,
        so it never triggers a Dynamo graph break, unlike ``_chunk_fully_masked``.

        Derivation: in this region only ``ctx_keep`` can be true. Its lower
        bound on ``j`` (``context_len - window_left + i``) increases with
        ``i``, so the union of keepable ``j`` over ``i`` in ``[q_start, q_end)``
        is widest at ``i = q_start``: ``[context_len - window_left + q_start,
        context_len - 1]``. The chunk is fully masked iff that range doesn't
        intersect ``[kv_start, kv_end - 1]``.
        """
        lo = max(context_len - window_left + q_start, kv_start)
        hi = min(context_len - 1, kv_end - 1)
        return lo > hi

    @staticmethod
    def _causal_chunk_fully_masked_analytic(q_start, q_end, kv_start, kv_end, prefix_len, window_left):
        """Host-integer equivalent of ``_chunk_fully_masked`` for a causal-loop
        chunk (``kv_start >= prefix_len``) in the band-mask (memory-win) path.

        Bit-exact vs. the materialized ``keep`` grid (verified by brute-force +
        exhaustive sweep, including window_left <= 0, which degenerates to
        "always fully masked" and is guarded explicitly below since it isn't
        captured by the two interval inequalities on its own).

        Derivation: in this region only ``cur_keep`` can be true: with
        ``q = j - prefix_len``, keep iff ``q <= i <= q + window_left - 1``.
        The chunk keeps something iff the query range ``[q_start, q_end)``
        intersects the union over ``q`` in ``[kv_start - prefix_len,
        kv_end - prefix_len)`` of ``[q, q + window_left - 1]``, i.e. iff
        ``kv_start - prefix_len <= q_end - 1`` and
        ``q_start <= kv_end - prefix_len + window_left - 2``.
        """
        if window_left <= 0:
            return True
        qs = kv_start - prefix_len
        qe = kv_end - prefix_len
        not_masked = (qs <= q_end - 1) and (q_start <= qe + window_left - 2)
        return not not_masked

    @staticmethod
    def _window_band_mask(q_start, q_end, kv_start, kv_end, prefix_len, context_len, window_left, dtype, device,
                          ndim=4):
        """Build a small sliding-window bias on the fly, replicating a slice of
        the full window mask that ``_set_attn_bias_for_sliding_window`` would
        otherwise materialize.

        This is the sliding-window memory win: rather than allocating the whole
        ``[batch, 1, seq, context+seq]`` mask, each surviving chunk gets only
        its own ``[q_chunk, kv_chunk]`` band. ``ndim`` selects the broadcast
        shape: 4 -> ``[1, 1, q, kv]`` for the plain layout, 5 ->
        ``[1, 1, 1, q, kv]`` to match GQA-reshaped inputs.

        Layout (must match the caller exactly):
          - key index ``j in [0, prefix_len)`` is the *padded* context region;
            ``prefix_len`` equals the caller's ``max_context_len``.
          - key index ``j in [prefix_len, kv_len)`` is the current query region,
            i.e. current query position ``q = j - prefix_len``.
          - ``context_len`` is the *valid* (unpadded) context length; the window
            origin and the padded-context masking are both measured from it.

        Keep (bias 0), else -inf:
          - context region: ``context_len - window_left + i <= j <= context_len - 1``
            (window lower edge + valid-context upper edge; masks padded context).
          - query region: ``i - window_left + 1 <= q <= i`` (window + causal).
        Mirrors ``_set_attn_bias_for_sliding_window``'s block-list branch, which
        uses ``invalid_lens = context_len - window + i - 1`` and a
        tril/triu(diagonal=-window+1) causal band.
        """
        i = torch.arange(q_start, q_end, device=device).view(-1, 1)
        j = torch.arange(kv_start, kv_end, device=device).view(1, -1)
        ctx_keep = (j >= (context_len - window_left + i)) & (j <= (context_len - 1)) & (j < prefix_len)
        q = j - prefix_len
        cur_keep = (q >= (i - window_left + 1)) & (q <= i) & (j >= prefix_len)
        keep = ctx_keep | cur_keep
        # Use a large *finite* negative (not -inf): a chunk / query row can be
        # entirely out-of-window under slicing, and -inf there would poison the
        # online-softmax rescaling with NaN (exp(-inf - -inf)). This matches the
        # finite fill the full-mask builder uses for the same reason.
        off_value = torch.finfo(dtype).min
        bias = torch.zeros((q_end - q_start, kv_end - kv_start), dtype=dtype, device=device)
        bias = bias.masked_fill(~keep, off_value)
        shape = (1, ) * (ndim - 2) + (q_end - q_start, kv_end - kv_start)
        return bias.view(*shape)

    def _chunked_attention(self, q, k, v, attn_mask, dropout_p, scale, softmax_mode, chunk_kernel_fn,
                           window_size=None, context_len=None):
        """Run chunked attention with online softmax rescaling.

        Args:
            q, k, v: Query, key, value tensors (after GQA reshape if needed).
            attn_mask: Attention mask tensor encoding causality, right padding
                and (for sliding-window layers) the window band. May be ``None``
                only in the sliding-window memory-win path (see ``context_len``),
                where per-chunk band masks are generated on the fly.
            dropout_p: Dropout probability.
            scale: Attention scale factor.
            softmax_mode: Softmax mode string.
            chunk_kernel_fn: Callable
                ``(q, k, v, mask, dropout_p, scale, is_causal, softmax_mode)``
                returning ``(out, m, linv)`` all as float32.
            window_size: Optional ``(left, right)`` sliding-window size. When set
                (with ``right == 0``, the causal case), KV chunks whose mask
                slice is entirely masked off -- i.e. the whole chunk falls
                outside the sliding window -- contribute nothing to the online
                softmax and are skipped, saving prefill compute.
            context_len: Valid (unpadded) context length. Required when
                ``attn_mask is None`` under a window: the full
                ``[batch, 1, seq, ctx+seq]`` mask is not materialized and each
                surviving chunk instead builds its own small band mask from
                ``context_len`` + ``window_size`` (the sliding-window memory win).

        Returns:
            Concatenated output tensor in float32.
        """
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        prefix_len = kv_len - q_len

        windowed = window_size is not None
        window_left = window_size[0] if windowed else None
        # Memory-win path: no full mask given, generate band masks on the fly.
        gen_band = windowed and attn_mask is None
        if gen_band:
            assert context_len is not None, 'context_len is required to generate band masks without a full mask'

        def chunk_mask(q_start, q_end, kv_start, kv_end, need_mask):
            if gen_band:
                return self._window_band_mask(q_start, q_end, kv_start, kv_end, prefix_len, context_len, window_left,
                                              q.dtype, q.device, ndim=q.dim())
            if need_mask:
                return attn_mask[..., q_start:q_end, kv_start:kv_end].contiguous()
            return None

        chunk_outputs = []
        num_q_chunks = math.ceil(q_len / self.chunk_size)
        num_prefix_chunks = math.ceil(prefix_len / self.chunk_size)
        for q_chunk_idx in range(num_q_chunks):
            q_start = q_len - (q_chunk_idx + 1) * self.chunk_size
            q_start = max(q_start, 0)
            q_end = q_len - q_chunk_idx * self.chunk_size
            q_chunk_size = q_end - q_start
            q_chunk = q[..., q_start:q_end, :].contiguous()

            last_out = None
            last_m = None
            last_linv = None

            # the causal part
            for kv_chunk_idx in range(0, num_q_chunks - q_chunk_idx):
                kv_start = prefix_len + q_end - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, prefix_len)
                kv_end = prefix_len + q_end - kv_chunk_idx * self.chunk_size

                # Sliding window: a non-diagonal causal chunk fully outside the
                # window for every query in this q-chunk contributes nothing.
                # The diagonal chunk (kv_chunk_idx == 0) always holds each
                # query's own position, so it is never fully masked.
                #
                # In the band-mask (memory-win) path the skip is decided from
                # host ints via _causal_chunk_fully_masked_analytic *before*
                # the mask is built, so a skipped chunk never even allocates a
                # band-mask tensor and the decision never touches the device
                # (no bool(tensor) sync, no Dynamo graph break). The mask-based
                # _chunk_fully_masked fallback below is for the (non-production)
                # full-mask window path, where the mask already exists.
                mask_chunk = None
                if windowed and kv_chunk_idx != 0:
                    if gen_band:
                        skip = self._causal_chunk_fully_masked_analytic(q_start, q_end, kv_start, kv_end, prefix_len,
                                                                        window_left)
                    else:
                        mask_chunk = chunk_mask(q_start, q_end, kv_start, kv_end, True)
                        skip = self._chunk_fully_masked(mask_chunk)
                    if skip:
                        self.maybe_break_graph()
                        continue

                # Always pass explicit mask for the diagonal chunk (kv_chunk_idx==0)
                # to ensure numerical consistency. The kernel's is_causal=True path
                # uses a different internal algorithm that can diverge from the
                # explicit mask path even when both encode the same triangular pattern.
                # For non-diagonal chunks within the padded region, also pass mask.
                # Under a window, every surviving chunk needs its mask so the
                # window edge is applied.
                if mask_chunk is None:
                    need_mask = (windowed or kv_chunk_idx == 0 or kv_chunk_idx < self.num_padded_query_chunks)
                    mask_chunk = chunk_mask(q_start, q_end, kv_start, kv_end, need_mask)

                kv_chunk_size = kv_end - kv_start
                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()

                self.maybe_break_graph()

                chunk_out, chunk_m, chunk_linv = chunk_kernel_fn(q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p,
                                                                 scale, False, softmax_mode)

                last_out, last_m, last_linv = self._merge_chunk(last_out, last_m, last_linv, chunk_out, chunk_m,
                                                                chunk_linv)

                self.maybe_break_graph()

            # the context part
            # Tight per-batch padded-chunk bound: when context_len (real max
            # context, in tokens) is available, only the tail chunks covering
            # [context_len, prefix_len) contain padding -- compute exactly how
            # many that is instead of using the global worst-case scalar.
            # Falls back to self.num_padded_ctx_chunks when context_len is
            # unknown (non-gen_band path).
            if context_len is not None and context_len < prefix_len:
                _num_padded_ctx_chunks = math.ceil((prefix_len - context_len) / self.chunk_size)
            else:
                _num_padded_ctx_chunks = self.num_padded_ctx_chunks

            for kv_chunk_idx in range(num_prefix_chunks):
                kv_start = prefix_len - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, 0)
                kv_end = prefix_len - kv_chunk_idx * self.chunk_size

                # Entire context chunk outside the sliding window for every
                # query in this q-chunk contributes nothing to the online
                # softmax, so skip it (saves prefill compute). See the causal
                # loop above for why the band-mask path decides this from host
                # ints (no mask built, no device sync, no graph break) while
                # the full-mask path still needs the mask tensor.
                mask_chunk = None
                if windowed:
                    if gen_band:
                        skip = self._context_chunk_fully_masked_analytic(q_start, q_end, kv_start, kv_end, prefix_len,
                                                                         context_len, window_left)
                    else:
                        mask_chunk = chunk_mask(q_start, q_end, kv_start, kv_end, True)
                        skip = self._chunk_fully_masked(mask_chunk)
                    if skip:
                        self.maybe_break_graph()
                        continue

                # use mask for chunks that may have padding; under a window also
                # pass the mask for surviving chunks so the window edge applies.
                if mask_chunk is None:
                    need_mask = windowed or kv_chunk_idx < _num_padded_ctx_chunks
                    mask_chunk = chunk_mask(q_start, q_end, kv_start, kv_end, need_mask)

                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()

                self.maybe_break_graph()

                chunk_out, chunk_m, chunk_linv = chunk_kernel_fn(q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p,
                                                                 scale, False, softmax_mode)

                assert not (last_out is None or last_m is None or last_linv is None)
                last_out, last_m, last_linv = self._merge_chunk(last_out, last_m, last_linv, chunk_out, chunk_m,
                                                                chunk_linv)

                self.maybe_break_graph()
            chunk_outputs.append(last_out)
        chunk_outputs = list(reversed(chunk_outputs))
        return torch.cat(chunk_outputs, dim=-2)


class SlicedFusedSDPA(SlicedFusedSDPABase):
    """Standalone module for BF16 sliced FusedSDPA.

    Extracting the sliced attention path into its own ``nn.Module`` allows it
    to be wrapped with ``torch.compile``, ``ht.hpu.wrap_in_hpu_graph``, or
    any other module-level wrapper independently of the dispatch logic in
    :class:`ModuleFusedSDPA`.
    """

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode, window_size=None,
                context_len=None):
        # attn_mask may be None only in the sliding-window memory-win path, where
        # per-chunk band masks are built on the fly from context_len.
        assert is_causal and (attn_mask is not None or (window_size is not None and context_len is not None))

        from habana_frameworks.torch.hpex.kernels.FusedSDPA import is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        gqa = is_gqa(query, key)
        if gqa:
            if attn_mask is not None:
                q, k, v, attn_mask = gqa_input_reshape_fwd(query, key, value, attn_mask)
            else:
                # No full mask to reshape; band masks are [1,1,q,kv] and
                # broadcast over the reshaped GQA head groups.
                q, k, v, _ = gqa_input_reshape_fwd(query, key, value, None)
        else:
            q, k, v, attn_mask = (query, key, value, attn_mask)
        if scale is None:
            scale = 1.0 / (query.shape[-1]**0.5)

        def chunk_kernel(q_c, k_c, v_c, mask_c, dp, sc, is_c, sm):
            res = torch.ops.hpu.sdpa_recomp_fwd(q_c, k_c, v_c, mask_c, dp, sc, is_c, True, sm, None, 'right')
            out, m, linv = tuple((gqa_output_reshape(x) if gqa else x).to(torch.float32) for x in res[:3])
            return out, m, linv

        output = self._chunked_attention(q, k, v, attn_mask, dropout_p, scale, softmax_mode, chunk_kernel,
                                         window_size=window_size, context_len=context_len)
        return output.to(q.dtype)


class ModuleFusedSDPA(torch.nn.Module):

    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'fusedSDPA kernel is None'
        self._hpu_kernel_fsdpa = fusedSDPA
        self._sliced_module = SlicedFusedSDPA()

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
        window_size=None,
        sinks=None,
        context_len=None,
    ):
        # Preconditions shared by both the non-window and the sliding-window
        # sliced paths.
        base_ok = (self._sliced_module.enable_slicing
                   and query.shape[0] == 1  # bs should be 1 for prefix-prefill
                   and query.shape[-2] != key.shape[-2]  # normal prefill (q_len == kv_len) routes to default
                   and is_causal  # only causal attention
                   and padding_side == 'right'  # right padding only for the chunks that may have padding
                   and sinks is None)  # not compatible with kernel fusion with sinks

        window_slicing = (window_size is not None and getattr(self._sliced_module, '_enable_window_slicing', False)
                          and window_size[1] == 0)  # opt-in, causal window only

        if window_slicing and base_ok:
            prefix_len = key.shape[-2] - query.shape[-2]
            will_skip = window_slicing_will_skip_chunks(prefix_len, self._sliced_module.chunk_size, window_size[0])
            if will_skip and (attn_mask is not None or context_len is not None):
                logger().debug_once(f'Sliced FusedSDPA handling sliding-window prefill '
                                    f'(window_size={window_size}, chunk skip enabled, '
                                    f'band_masks={attn_mask is None}).')
                return self._sliced_module(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                           window_size=window_size, context_len=context_len)
        elif window_size is None and base_ok and attn_mask is not None \
                and key.shape[-2] >= self._sliced_module.slice_thld:  # apply for kv_len >= slice_thld only
            return self._sliced_module(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode)

        if is_causal and attn_mask is not None:
            # TODO: causal + attn_bias is not yet supported
            is_causal = False
            valid_sequence_lengths = None

        if window_size is not None:
            return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                                recompute_mode, valid_sequence_lengths, padding_side, False, False,
                                                window_size, sinks)
        else:
            return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                                recompute_mode, valid_sequence_lengths, padding_side, False, False,
                                                (-1, -1), sinks)


class SlicedFP8FusedSDPA(SlicedFusedSDPABase):
    """Standalone module for FP8 sliced FusedSDPA.

    Like :class:`SlicedFusedSDPA`, extracting the sliced path enables
    wrapping with ``torch.compile`` or ``ht.hpu.wrap_in_hpu_graph``.
    Expects pre-quantized FP8 inputs; dequantises chunk outputs to
    BF16/FP32 before the online-softmax rescaling merge.
    """

    def __init__(self, parent):
        super().__init__()
        # Store parent reference without registering as a submodule
        # to avoid circular module graph while sharing scale tensors.
        object.__setattr__(self, '_parent', parent)

    def _dequant_output(self, output):
        return torch.ops.hpu.cast_from_fp8(output, self._parent.d_scale_output, torch.bfloat16)

    def _fp8_fsdpa_fwd(self, q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode):
        results = torch.ops.hpu.fp8_sdpa_recomp_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            True,  # requires_backward
            softmax_mode,
            self._parent.d_scale_q,
            self._parent.d_scale_k,
            self._parent.d_scale_v,
            self._parent.scale_amax,
            self._parent.d_scale_output,
            self._parent.descale_amax,
            False,  # is_amax_s
            False,  # is_amax_o
            None,  # valid_seq_len
            "right",  # padding_side
            (-1, -1),  # window_size
            None,  # sinks
        )
        return results

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
        assert is_causal and attn_mask is not None

        from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        gqa = is_gqa(query, key)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(query, key, value, attn_mask)
        else:
            q, k, v, attn_mask = (query, key, value, attn_mask)
        softmax_mode = softmax_mode if softmax_mode == "fp32" else "fast"
        if scale is None:
            scale = 1.0 / (query.shape[-1]**0.5)

        def chunk_kernel(q_c, k_c, v_c, mask_c, dp, sc, is_c, sm):
            res = self._fp8_fsdpa_fwd(q_c, k_c, v_c, mask_c, dp, sc, is_c, sm)
            out, m, linv = tuple(gqa_output_reshape(x) if gqa else x for x in res[:3])
            m = m.to(torch.float32)
            linv = linv.to(torch.float32) * (128.0 if sm == "fast" else 1.0)
            out = self._dequant_output(out).to(torch.float32)
            return out, m, linv

        return self._chunked_attention(q, k, v, attn_mask, dropout_p, scale, softmax_mode, chunk_kernel)


class ModuleFP8FusedSDPA(torch.nn.Module):

    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'FP8 fusedSDPA kernel is None'
        self.fp8_fused_sdpa = fusedSDPA

        # set the descale_amax and scale_amax 1.0 temporarily
        self.descale_amax = torch.tensor(1.0, dtype=torch.float32)
        self.scale_amax = torch.tensor(1.0, dtype=torch.float32)
        self.scale_q = torch.tensor(1.0, dtype=torch.float32)
        self.scale_k = torch.tensor(1.0, dtype=torch.float32)
        self.scale_v = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_q = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_k = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_v = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_output = torch.tensor(1.0, dtype=torch.float32)
        self._sliced_module = SlicedFP8FusedSDPA(parent=self)

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, torch.float8_e4m3fn)[0]

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
        window_size=None,
        context_len=None,  # accepted for signature parity; FP8 has no window slicing
    ):

        qinput = self.quant_input(query, self.scale_q).detach()
        kinput = self.quant_input(key, self.scale_k).detach()
        vinput = self.quant_input(value, self.scale_v).detach()

        bs = query.shape[0]
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        if (self._sliced_module.enable_slicing and kv_len >= self._sliced_module.slice_thld \
                and bs == 1  # bs should be 1 for chunked prefill
                and q_len != kv_len  # normal causal prefill route to the default dispatch for better performance
                and is_causal and attn_mask is not None  # only supports causal attention with mask
                and padding_side == 'right'  # currently only supports right padding for the chunks that may have padding
                and window_size is None  # slicing is not compatible with sliding window attention
            ):
            return self._sliced_module(qinput, kinput, vinput, attn_mask, dropout_p, is_causal, scale,
                                       softmax_mode).to(query.dtype)

        if is_causal and attn_mask is not None:
            # TODO: causal + attn_bias is not yet supported
            is_causal = False
            valid_sequence_lengths = None

        results = self.fp8_fused_sdpa(
            qinput,
            kinput,
            vinput,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            softmax_mode=softmax_mode,
            d_scale_q=self.d_scale_q,
            d_scale_k=self.d_scale_k,
            d_scale_v=self.d_scale_v,
            q_scale_s=self.scale_amax,
            # q_scale_o=1 / 1.0,
            d_scale_s=self.descale_amax,
            is_amax_s=False,
            valid_seq_len=valid_sequence_lengths,
            seq_padding_type=padding_side,
        )

        output = results[0]
        return output


def pad_list(input, target_len, val_generator):
    padding = target_len - len(input)
    if padding > 0:
        input.extend(itertools.islice(val_generator, padding))
    return input


def align_and_pad(data, bucketing, padding_gen):
    bs = len(data)
    target_bs, target_len = bucketing
    if target_bs == 1 and bs > 1:
        data = [list(itertools.chain(*data))]
    data = [pad_list(x, target_len, padding_gen) for x in data]
    padding = itertools.islice(padding_gen, target_len)
    data = pad_list(data, target_bs, itertools.tee(padding, target_bs - len(data)))
    return data


def with_default(value: Optional[Any], default: Any) -> Any:
    if value is not None:
        return value
    return default
