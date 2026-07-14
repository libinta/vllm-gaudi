# Option C — Window-aware SlicedFusedSDPA: Implementation Kickoff

> Hand-off spec for a fresh session. Read this first, then confirm the current
> code matches the line references (they drift). Work on branch
> `libinta/gemma4-sliding-window-kv` (or a fresh branch off `libinta/enable_gemma4`).

## What C is (and is NOT)

**C saves prefill *mask memory* + prefill *compute* for sliding-window layers.**
It makes the chunked FusedSDPA path window-aware: skip KV chunks that fall
entirely outside the sliding window, and never materialize the full
`[batch, 1, seq, ctx+seq]` attention mask.

**C does NOT free KV cache blocks.** Sliding layers still allocate full-length
KV (they stay `FullAttentionSpec`, one KV group, one block table, one
slot_mapping). The "fit 128k / higher concurrency" KV-capacity win is Option A,
a separate, much larger change. Do not conflate them.

**Why C is easy:** it is entirely inside `vllm_gaudi/extension/utils.py`
(the `SlicedFusedSDPA` / `ModuleFusedSDPA` classes). No KV cache groups, no
per-group block tables, no slot_mapping changes, no `hpu_attn.py` metadata
plumbing. That is the whole reason to prefer it.

## Environment (verified working this session)

- Gaudi3 container `litang_482`. Source is bind-mounted: host
  `/home/litang/github/gemma4/vllm-gaudi` == container
  `/root/litang/github/gemma4/vllm-gaudi` (edit on host, runs in container).
- Run tests / models inside the container:
  `docker exec litang_482 bash -lc '... '`
- Toolchain (torch 2.11, habana) only works INSIDE the container; the host
  cannot import `vllm_gaudi`. `pytest` is installed in the container.
- `PYTHONPATH=/root/litang/github/gemma4/vllm:/root/litang/github/gemma4/vllm-gaudi`
- HF cache: `export HF_HOME=/software/data/pytorch/huggingface`
- Pin cards with `HABANA_VISIBLE_MODULES="4,5"` (check `hl-smi` for free cards;
  each TP=2 run needs 2 free cards).

## CRITICAL harness lesson (cost this session hours)

Instruction-tuned Gemma produces **degenerate garbage** ("la la la") with raw
prompts. You MUST use the chat template (adds BOS + turn formatting). Use
`llm.chat([{"role":"user","content":...}], sp)`, NOT `llm.generate("raw text")`.
With the chat template, `google/gemma-4-31B-it` answers "What is 12x8?" -> "96".
That coherent baseline is your correctness reference.

Fast validation: set `VLLM_SKIP_WARMUP=true` (warmup is ~10 min/run and does
NOT change output correctness — verified). Use `google/gemma-4-31B-it` (dense,
non-MoE) to avoid the separate MoE `_maybe_pad_hidden_states` bug. Smoke harness
lives at `/home/litang/github/gemma4/test/smoke_sliding_window_kv.py` (already
uses chat template + skip-warmup friendly).

## Where the code is (verify line numbers before editing)

`vllm_gaudi/extension/utils.py`:
- `SlicedFusedSDPABase._chunked_attention(...)` ~line 271 — the chunked loop
  with online-softmax rescaling. Two inner loops:
  - "causal part" (~line 306): iterates current-chunk KV.
  - "context part" (~line 333): iterates ALL prefix chunks — **this is where
    out-of-window chunks must be skipped for sliding window.**
- `SlicedFusedSDPA.forward(...)` ~line 367 — asserts `is_causal and attn_mask
  is not None`; wraps `_chunked_attention`.
- `ModuleFusedSDPA.forward(...)` ~line 396 — dispatch. Line ~418 currently
  DISABLES slicing when `window_size is not None`:
  `and window_size is None  # slicing is not compatible with sliding window`.
  That gate is what C removes/relaxes.

Caller context (read-only, to understand what reaches the kernel):
- `vllm_gaudi/attention/backends/hpu_attn.py` prefill branch ~line 627: when
  `self.sliding_window`, it either passes `window_attn_bias` (explicit mask) or
  sets `common_args['window_size'] = (sliding_window, 0)`.
- The explicit prefill mask is built in
  `hpu_model_runner.py::HPUAttentionMetadataProcessor._set_attn_bias_for_sliding_window`
  (~line 7132 on the A branch; find by name) — shape
  `[batch,1,seq, max_context_len + seq]`. Eliminating the need to materialize
  this at long context is C's memory win.

## Implementation steps (TDD, commit per step)

### Step 0: Baseline reference
Run the smoke harness flag-agnostic (C has its own trigger, see Step 3) on
`gemma-4-31B-it` with chat template + skip warmup. Capture coherent output
("96", real sentences) as the correctness bar. Also capture peak prefill memory
if you can (for the win metric).

### Step 1: Understand the window semantics
In `_chunked_attention`, `q` positions and `k` positions are known. For a
`(sliding_window, 0)` window, query position `i` attends to key positions
`(i - sliding_window, i]`. A whole KV chunk `[kv_start, kv_end)` is entirely
OUTSIDE the window for a query chunk `[q_start, q_end)` iff
`kv_end <= (q_start_abs - sliding_window + 1)` where positions are absolute
(account for `prefix_len`). Those chunks contribute nothing to online softmax
and can be `continue`-skipped.

### Step 2: Make `_chunked_attention` window-aware
Add an optional `window_size` param (default None). In the context-part loop
(~line 333), skip KV chunks fully outside the window. For the single boundary
chunk that straddles the window edge, generate a small `[q_chunk, kv_chunk]`
band mask on the fly instead of slicing a materialized full mask. The
`_merge_chunk` online-softmax math is unchanged (skipped chunks add nothing).
Unit-test the skip predicate + boundary mask in pure PyTorch on CPU-in-container
(shape/values), mirroring `tests/unit_tests/test_sliding_window_kv_spec.py`
style (pure-logic tests, no HPU).

### Step 3: Ungate slicing for window in `ModuleFusedSDPA.forward`
Remove/relax the `window_size is None` condition (~line 418) and thread
`window_size` into `self._sliced_module(...)`. Keep a flag/guard so this only
activates intentionally (e.g. reuse the existing slicing enable + a check that
window_size is set). Preserve all other slicing preconditions (bs==1,
kv_len>=slice_thld, right padding, sinks is None).

### Step 4: Validate on device
Run smoke harness on `gemma-4-31B-it`, chat template, with a prompt long enough
to exceed `slice_thld` and trigger the sliced path (short prompts won't).
Compare output to the Step 0 baseline — MUST match (window math is exact;
positions outside the window contribute ~0). Confirm no crash, coherent text.

### Step 5: Measure the win
Show prefill peak memory / that the full `[seq, ctx+seq]` mask is no longer
materialized (or is much smaller) at long context. Document before/after.

## Correctness bar
Output with C enabled MUST match the flag-off baseline on the SAME prompts
(chat template, temperature=0). Sliding-window attention is mathematically
exact up to the window; C only avoids computing/masking positions that are
already -inf. Any divergence = a bug in the skip predicate or boundary mask.

## Do NOT
- Do NOT touch KV cache specs, groups, block tables, or slot_mapping — that is
  Option A and out of scope for C.
- Do NOT test with raw `llm.generate` prompts — use the chat template.
- Do NOT trust warmed-vs-skip-warmup to differ in correctness — it does not;
  use skip-warmup for speed.

## Reference
Full architectural analysis of A vs C and the sliding-window investigation is
in the conversation that produced `2026-07-13-gemma4-sliding-window-kv-spec.md`
(Option A plan) on this branch.
