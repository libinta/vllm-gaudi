# Gemma4 Sliding-Window KV Cache (Option A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Gemma4's sliding-window attention layers allocate KV cache only for their window (~1024 tokens) instead of the full context (~120k), by emitting `SlidingWindowSpec` for those layers so vLLM's hybrid KV cache manager evicts out-of-window blocks — freeing HBM to fit full 128k context and multiply concurrency.

**Architecture:** vLLM core already supports this end-to-end: `Attention.get_kv_cache_spec()` returns `SlidingWindowSpec` for sliding layers, `HybridKVCacheCoordinator` + `SlidingWindowManager` manage per-group eviction, and `MultiGroupBlockTable` tracks per-group block ids. The HPU platform already returns `support_hybrid_kv_cache() == True`, so nothing re-unifies the spec. The gap is entirely in vllm-gaudi's `HPUModelRunner`: (1) `get_kv_cache_spec` hardcodes `FullAttentionSpec` for every decoder layer instead of delegating to the module; (2) the decode/prefill paths select the attention block table by hardcoded group index `0` or by mamba-only heuristics; (3) the prefill sliding-window mask assumes context columns map to absolute position 0, which breaks once old blocks are evicted. This plan closes those three gaps behind one gaudi env flag, with no vllm-core edits.

**Tech Stack:** Python, PyTorch, vllm-gaudi (`vllm_gaudi`), vLLM v1 core (consumed as-is), pytest, HPU (Gaudi3).

## Global Constraints

- **No vllm-core edits.** All changes live under `vllm_gaudi/`. Core classes (`SlidingWindowSpec`, `SlidingWindowManager`, `HybridKVCacheCoordinator`, `MultiGroupBlockTable`, `Attention.get_kv_cache_spec`) are consumed, never modified.
- **Feature flag:** `VLLM_HPU_SLIDING_WINDOW_KV` — default `"0"` (off) during bring-up. When off, behavior is byte-for-byte identical to today (all `FullAttentionSpec`, single group). Follow the GDN idiom at `hpu_model_runner.py:1158-1180` (auto-set default, user-overridable, auto-disable on incompatibility).
- **Correctness gate:** the feature must NOT be enable-able together with an active KV connector that lacks HMA support, or eagle+chunked-local-attention. Auto-disable and log, mirroring `hpu_model_runner.py:1166-1174`.
- **Python line length:** match the file (this file uses long lines; no hard 88 cap enforced here — match surrounding code).
- **Target model:** `google/gemma-4-26B-A4B-it` — 30 layers, `layer_types` = 25 `sliding_attention` + 5 `full_attention`, `sliding_window=1024`, sliding head_dim=256/8 kv-heads, full head_dim=512/2 kv-heads, TP=2, `enable_prefix_caching=True`, `enable_chunked_prefill=True`, `block_size=128`.
- **Heterogeneous head sizes must keep working:** the existing normalization comment at `hpu_model_runner.py:1522-1531` (sliding head_size=256 vs full head_size=512) must be preserved; do not regress it.

---

## File Structure

- `vllm_gaudi/v1/worker/hpu_model_runner.py` — the only production file modified. Changes are localized to:
  - `__init__` / config block (~line 1195): add flag + auto-disable logic → **Task 1**
  - `get_kv_cache_spec` (~lines 1485-1554): emit `SlidingWindowSpec` per layer → **Task 2**
  - `_get_attention_group_id_for_hybrid` + decode/prefill block-table selection (~lines 2373, 6743, 6785) + `_create_decode_input_data` window path (~line 2968) → **Task 3**
  - bucketing for the sliding group (`get_habana_paged_attn_buffers` call sites) → **Task 4**
  - `_set_attn_bias_for_sliding_window` prefill mask (~lines 7043-7110) → **Task 5**
- `tests/unit_tests/test_sliding_window_kv_spec.py` — new unit tests (Tasks 1, 2, 5).
- `tests/unit_tests/test_decode_bucket_hybrid.py` — existing; extend for sliding group (Task 4).

Tasks are ordered so each is independently testable. Tasks 1-2 produce the spec (verifiable in isolation with a mocked config). Task 3-4 wire the runtime (verifiable via decode-input construction). Task 5 fixes correctness (verifiable via mask-shape/value assertions). Task 6 is the end-to-end verification on-device.

---

### Task 1: Add the `VLLM_HPU_SLIDING_WINDOW_KV` flag with auto-disable logic

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py` (config block near line 1195, and the flag read near where `interleaved_sliding_window` is set, line ~1070)
- Test: `tests/unit_tests/test_sliding_window_kv_spec.py` (create)

**Interfaces:**
- Produces: `self.use_sliding_window_kv: bool` — attribute on `HPUModelRunner`, True only when the flag is on, the model is interleaved-SWA, the hybrid manager is enabled, and no incompatible connector/eagle config is present. Read by Task 2 and Task 3.
- Produces: module-level helper `_sliding_window_kv_enabled(vllm_config) -> bool` for unit-testing the gating logic without constructing a full runner.

- [ ] **Step 1: Write the failing test**

Create `tests/unit_tests/test_sliding_window_kv_spec.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Gemma4 sliding-window KV cache spec (Option A)."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_gaudi.v1.worker.hpu_model_runner import _sliding_window_kv_enabled


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
    # is_interleaved is checked inside; patch it to True for this SWA model.
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -v`
Expected: FAIL with `ImportError: cannot import name '_sliding_window_kv_enabled'`

- [ ] **Step 3: Write minimal implementation**

Add near the top-level helpers of `hpu_model_runner.py` (after the imports / other module-level helpers, e.g. right before `class HPUModelRunner` or near `compute_prefix_caching_block_indices` at line ~672):

```python
def _sliding_window_kv_enabled(vllm_config) -> bool:
    """Gate for Option A: emit SlidingWindowSpec for interleaved-SWA models.

    Off by default. Requires VLLM_HPU_SLIDING_WINDOW_KV=1, an interleaved
    sliding-window model, the hybrid KV cache manager enabled, and no
    incompatible connector / eagle+chunked-local-attention config.
    """
    if os.getenv("VLLM_HPU_SLIDING_WINDOW_KV", "0").strip().lower() \
            not in ("1", "true"):
        return False
    hf_text = vllm_config.model_config.hf_text_config
    if not (is_interleaved(hf_text)
            and getattr(hf_text, "sliding_window", None)):
        return False
    # Core must not be unifying specs back to FullAttentionSpec.
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        return False
    # eagle + chunked local attention is unsupported by the hybrid manager.
    spec_cfg = vllm_config.speculative_config
    if (getattr(vllm_config.model_config, "attention_chunk_size", None)
            is not None and spec_cfg is not None
            and getattr(spec_cfg, "use_eagle", lambda: False)()):
        return False
    return True
```

Then in `HPUModelRunner.__init__` (near line 1195, alongside `self.use_prefix_caching`), add:

```python
        self.use_sliding_window_kv = _sliding_window_kv_enabled(vllm_config)
        if self.use_sliding_window_kv:
            logger.info(
                "Sliding-window KV cache (Option A) ENABLED: sliding layers "
                "will allocate KV only within the %d-token window.",
                self.sliding_window)
```

Confirm `import os` and `from vllm.transformers_utils.config import is_interleaved` are already present (they are — line 94).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/litang/github/gemma4/vllm-gaudi
git add tests/unit_tests/test_sliding_window_kv_spec.py vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "feat(hpu): add VLLM_HPU_SLIDING_WINDOW_KV gate for Gemma4 sliding-window KV"
```

---

### Task 2: Emit `SlidingWindowSpec` per layer in `get_kv_cache_spec`

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py:1485-1554` (`get_kv_cache_spec`)
- Modify: import block at `hpu_model_runner.py:71-86` (add `SlidingWindowSpec`)
- Test: `tests/unit_tests/test_sliding_window_kv_spec.py` (extend)

**Interfaces:**
- Consumes: `self.use_sliding_window_kv` (Task 1).
- Produces: `get_kv_cache_spec()` returns a dict where sliding layers map to `SlidingWindowSpec(sliding_window=...)` and full layers to `FullAttentionSpec`, **only** when `self.use_sliding_window_kv` is True; otherwise unchanged (all `FullAttentionSpec`).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/test_sliding_window_kv_spec.py`:

```python
from unittest.mock import MagicMock
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.attention.backend import AttentionType


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
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    runner = MagicMock(spec=HPUModelRunner)
    runner.use_sliding_window_kv = use_swa
    runner.kv_cache_dtype = dtype
    fwd_ctx = layers
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=fwd_ctx),
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype="auto"))
    runner.vllm_config = vllm_config
    runner.shared_kv_cache_layers = {}
    # Bind the real method to the stub:
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
```

Note: if `get_kv_cache_spec` touches attributes not set on the MagicMock, set them in `_spec_from_layers` to match the real method's reads (e.g. `runner.kv_cache_dtype`). Adjust the stub to whatever the final method body reads — keep the stub minimal.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -k "sliding_layer_emits or all_full_when" -v`
Expected: FAIL — `test_sliding_layer_emits_sliding_spec_when_enabled` asserts `SlidingWindowSpec` but current code returns `FullAttentionSpec`.

- [ ] **Step 3: Write minimal implementation**

Add to the import block (line 71-86, inside the `from vllm.v1.kv_cache_interface import (` group):

```python
    SlidingWindowSpec,
```

Replace the DECODER branch in `get_kv_cache_spec` (lines 1521-1535) with:

```python
                if attn_module.attn_type == AttentionType.DECODER:
                    # Normalize page_size_bytes across heterogeneous head_size layers
                    # (e.g. Gemma4: sliding head_size=256, full head_size=512)
                    max_head_size = max(
                        m.head_size for m in forward_ctx.values()
                        if isinstance(m, Attention)
                        and m.attn_type == AttentionType.DECODER
                        and getattr(m, 'kv_sharing_target_layer_name', None) is None
                    )
                    layer_block_size = block_size
                    if attn_module.head_size < max_head_size:
                        layer_block_size = block_size * (max_head_size // attn_module.head_size)
                    per_layer_sliding = getattr(attn_module, 'sliding_window', None)
                    if self.use_sliding_window_kv and per_layer_sliding:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=layer_block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                            sliding_window=per_layer_sliding)
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=layer_block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype)
```

Note: this also fixes the pre-existing dead `layer_block_size` bug (it was computed but never used at line 1531-1532; now it is applied). If applying `layer_block_size` to `FullAttentionSpec` changes today's behavior when the flag is OFF, revert the full branch to pass `block_size` (not `layer_block_size`) so OFF-path is byte-identical — verify against `test_all_full_when_disabled` and, if in doubt, keep `block_size=block_size` for `FullAttentionSpec` and only use `layer_block_size` for the new `SlidingWindowSpec` path.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
cd /home/litang/github/gemma4/vllm-gaudi
git add tests/unit_tests/test_sliding_window_kv_spec.py vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "feat(hpu): emit SlidingWindowSpec for Gemma4 sliding layers behind flag"
```

---

### Task 3: Select the correct attention block table per group (decode + prefill)

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py:2373-2380` (`_get_attention_group_id_for_hybrid`)
- Modify: `hpu_model_runner.py:6743`, `6785` (hardcoded `block_table[0]`)
- Modify: `hpu_model_runner.py:2968-2981` (decode window block table — must read the sliding group's own block table, not slice group 0)
- Test: manual/integration via Task 6; add a targeted unit test for the group-id helper.

**Interfaces:**
- Consumes: `self.use_sliding_window_kv`, `self.kv_cache_config.kv_cache_groups`.
- Produces: `self._get_full_attention_group_id() -> int` and `self._get_sliding_attention_group_id() -> int | None` — resolve group indices by spec type (`FullAttentionSpec` vs `SlidingWindowSpec`). Used wherever a block table was previously `block_table[0]` or `_get_attention_group_id_for_hybrid()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/test_sliding_window_kv_spec.py`:

```python
def test_group_id_resolution():
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    from vllm.v1.kv_cache_interface import (FullAttentionSpec,
                                            SlidingWindowSpec)
    runner = MagicMock(spec=HPUModelRunner)
    runner.use_sliding_window_kv = True
    full = SimpleNamespace(kv_cache_spec=MagicMock(spec=FullAttentionSpec))
    swa = SimpleNamespace(kv_cache_spec=MagicMock(spec=SlidingWindowSpec))
    # group order: full first (coordinator sorts full first), sliding second
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[full, swa])
    assert HPUModelRunner._get_full_attention_group_id(runner) == 0
    assert HPUModelRunner._get_sliding_attention_group_id(runner) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -k group_id -v`
Expected: FAIL — `AttributeError: _get_full_attention_group_id`.

- [ ] **Step 3: Write minimal implementation**

Add methods near `_get_attention_group_id_for_hybrid` (line 2373):

```python
    def _get_full_attention_group_id(self) -> int:
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        groups = self.kv_cache_config.kv_cache_groups
        for gid, group in enumerate(groups):
            if isinstance(group.kv_cache_spec, FullAttentionSpec):
                return gid
        # Fall back to the first attention group (single-group / all-full).
        return self._get_attention_group_id_for_hybrid()

    def _get_sliding_attention_group_id(self):
        from vllm.v1.kv_cache_interface import SlidingWindowSpec
        if not getattr(self, "use_sliding_window_kv", False):
            return None
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, SlidingWindowSpec):
                return gid
        return None
```

Then, in `_create_decode_input_data` (line 2960-2981), when `self.use_sliding_window_kv` and a sliding group exists, build the window block table from **that group's** block table rather than slicing group 0:

```python
        if self.interleaved_sliding_window:
            swa_gid = self._get_sliding_attention_group_id()
            if swa_gid is not None:
                # Option A: sliding layers have their own KV group; use its
                # (already window-bounded) block table directly.
                swa_bt = self.input_batch.block_table[swa_gid].get_cpu_tensor()
                window_block_tables = [
                    self._resolve_all_blocks(swa_bt[i, :n].tolist())
                    for i, n in enumerate(<per-req swa block counts>)
                ]
            else:
                # Legacy path: slice the single full-attention table.
                sliding_block_size = (self.sliding_window // decode_block_size)
                model_type = self._get_model_type()
                if model_type is not None and model_type in ["gpt_oss"]:
                    sliding_block_size += 1
                window_block_tables = [block_table[-sliding_block_size:]
                                       for block_table in block_tables_list]
            window_block_list, window_block_groups, window_block_usage = \
                self.get_habana_paged_attn_buffers(
                    window_block_tables, slot_mapping.tolist(),
                    padded_batch_size * num_tokens, block_size=decode_block_size)
```

`<per-req swa block counts>` = number of valid blocks per request in the sliding group's table; derive from `self.input_batch.num_computed_tokens_cpu` capped at the window: `min(cdiv(window-1+scheduled, decode_block_size)+1, cdiv(computed+scheduled, decode_block_size))`. Confirm the exact source when implementing against the real `input_batch` API; the intent is "the valid, window-bounded blocks the coordinator already allocated for the sliding group."

Replace the two hardcoded `self.input_batch.block_table[0]` reads at lines 6743 and 6785 with `self.input_batch.block_table[self._get_full_attention_group_id()]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -k group_id -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/litang/github/gemma4/vllm-gaudi
git add tests/unit_tests/test_sliding_window_kv_spec.py vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "feat(hpu): resolve full/sliding KV groups by spec type for decode block tables"
```

---

### Task 3b: Allocate KV tensors for `SlidingWindowSpec` layers

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py:6396-6438` (the non-hybrid `initialize_kv_cache` allocation loop — the path gemma4 takes, since it has no mamba layers)
- Test: covered by Task 6 Step 2 (startup must not crash); add a guard assertion.

**Interfaces:**
- Consumes: `SlidingWindowSpec` groups from Task 2.
- Produces: a `(key_cache, value_cache, key_scales, value_scales)` tuple allocated for each sliding layer, sized to that group's `num_blocks` — so `assert layer_names == set(kv_caches.keys())` (line 6451) passes.

**Background (why this is mandatory, not a risk):** the non-hybrid allocation loop branches on `isinstance(kv_cache_spec, FullAttentionSpec)` (line 6396) then `MambaSpec` (6430), else `raise ValueError` (6438). A `SlidingWindowSpec` matches neither → **startup crash**. `SlidingWindowSpec` is an `AttentionSpec` with the same `page_size_bytes` / `get_kv_cache_shape` contract as `FullAttentionSpec`, so it allocates identically — just fewer blocks (its `num_blocks` is smaller because its `page_size_bytes` per request is window-bounded via the hybrid config).

- [ ] **Step 1: Write the failing check**

There is no cheap CPU unit test for the full `initialize_kv_cache` (it allocates device tensors). Instead, add a startup guard and rely on Task 6 Step 2. First, reproduce the crash conceptually: with flag ON, `initialize_kv_cache` hits `raise ValueError(f"Unknown KV cache spec type ...")` for sliding layers. This step documents the expected failure.

Run (on device, flag ON — will crash before this task):
`VLLM_HPU_SLIDING_WINDOW_KV=1 bash test/run_enable_gemma4_482.sh`
Expected before fix: `ValueError: Unknown KV cache spec type for layer ... SlidingWindowSpec` (or an `assert layer_names == set(kv_caches.keys())` failure).

- [ ] **Step 2: Write minimal implementation**

Change the branch condition at line 6396 to accept both attention spec types. Add the import (line 71-86) already covers `SlidingWindowSpec` (Task 2). Then:

```python
                    if isinstance(kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec)):
                        kv_cache_shape = self.attn_backend.get_kv_cache_shape(num_blocks + 1, kv_cache_spec.block_size,
                                                                              kv_cache_spec.num_kv_heads,
                                                                              kv_cache_spec.head_size)
                        # ... (rest of the FullAttentionSpec allocation body unchanged) ...
```

Only the `isinstance(...)` guard changes — the entire allocation body (shape, scales, `kv_caches[layer_name] = (...)`) is identical because `SlidingWindowSpec` exposes the same `block_size`/`num_kv_heads`/`head_size`/`dtype`/`page_size_bytes` interface. Do the same for the two hybrid/mamba-path allocation branches **only if** a mamba+sliding model is in scope (it is not for gemma4 — skip to keep the change minimal, but note it for future SWA+mamba models).

- [ ] **Step 3: Verify startup succeeds**

Run (on device): `VLLM_HPU_SLIDING_WINDOW_KV=1 bash test/run_enable_gemma4_482.sh 2>&1 | grep -E "Usable num_blocks|ValueError|not correctly initialized"`
Expected: no `ValueError`, no assertion failure; `Usable num_blocks` line prints. (Full memory validation is Task 6.)

- [ ] **Step 4: Commit**

```bash
cd /home/litang/github/gemma4/vllm-gaudi
git add vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "feat(hpu): allocate KV tensors for SlidingWindowSpec layers"
```

---

### Task 4: Per-group bucketing for the sliding block list

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py` — `get_habana_paged_attn_buffers` call for the window list (line ~2977) and, if needed, the `_PAD_BLOCK_ID` used for contiguous PA.
- Test: `tests/unit_tests/test_decode_bucket_hybrid.py` (extend)

**Interfaces:**
- Consumes: `_get_sliding_attention_group_id()` (Task 3), the sliding group's block count.
- Produces: a window block list whose bucketing/padding is bounded by the sliding group's `max_num_blocks` (≈ `cdiv(window-1+max_num_batched_tokens, block)+1`), not the full pool's `max_blocks`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/test_decode_bucket_hybrid.py`:

```python
def test_sliding_window_bucket_bounded_by_window():
    """The sliding-group decode bucket must not exceed window-sized blocks."""
    block_size = 128
    sliding_window = 1024
    max_num_batched_tokens = 2048
    # Expected per-request sliding-group block cap (SlidingWindowSpec formula):
    expected_cap = math.ceil(
        (sliding_window - 1 + max_num_batched_tokens) / block_size) + 1
    # A window block list for bs=2 should never bucket beyond expected_cap*bs.
    from vllm_gaudi.v1.worker.hpu_model_runner import (
        _sliding_group_max_blocks_per_req)
    assert _sliding_group_max_blocks_per_req(
        sliding_window, block_size, max_num_batched_tokens) == expected_cap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_decode_bucket_hybrid.py -k sliding_window_bucket -v`
Expected: FAIL — `ImportError: _sliding_group_max_blocks_per_req`.

- [ ] **Step 3: Write minimal implementation**

Add a module-level helper (mirrors core `SlidingWindowSpec.max_admission_blocks_per_request`):

```python
def _sliding_group_max_blocks_per_req(sliding_window, block_size,
                                      max_num_batched_tokens):
    """Per-request block cap for a sliding-window KV group.

    Mirrors vllm SlidingWindowSpec.max_admission_blocks_per_request:
    hold the last (sliding_window-1) computed tokens plus the newly
    scheduled tokens, +1 for partial-block window start.
    """
    num_tokens = sliding_window - 1 + max_num_batched_tokens
    return math.ceil(num_tokens / block_size) + 1
```

Then use it to cap the bucket lookup for the window list. In the window branch of `_create_decode_input_data` (Task 3), pass a `max_blocks` hint bounded by `_sliding_group_max_blocks_per_req(...) * padded_batch_size` when calling `find_decode_bucket` for the window buffers, so contiguous-PA `block_bucket_size` and `_PAD_BLOCK_ID` don't inflate to the full-pool size. If `get_habana_paged_attn_buffers` derives the bucket internally from `max(block_list)`, the smaller sliding block ids already keep it bounded — in that case this helper documents/asserts the invariant and the test guards against regression.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_decode_bucket_hybrid.py -k sliding_window_bucket -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/litang/github/gemma4/vllm-gaudi
git add tests/unit_tests/test_decode_bucket_hybrid.py vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "feat(hpu): bound sliding-group decode bucket to window size"
```

---

### Task 5: Fix the prefill sliding-window mask to be window-relative

**Files:**
- Modify: `vllm_gaudi/v1/worker/hpu_model_runner.py:7073-7100` (`_set_attn_bias_for_sliding_window`, the `block_list is not None` branch)
- Test: `tests/unit_tests/test_sliding_window_kv_spec.py` (extend)

**Interfaces:**
- Consumes: the sliding group's prefill context length (`context_lens_tensor`) and its window block list (kept blocks × block_size).
- Produces: `past_mask` computed from **absolute** context positions `base + arange(K)` where `base = max(0, ceil(L/block_size)*block_size - K)`, so that when old blocks are evicted (Option A on), the mask still selects the correct in-window context columns instead of collapsing to all-False.

**Background (why this is the correctness gate):** today `past_indices = torch.arange(max_context_len)` assumes context column `c` is at absolute position `c`. Under Option A the sliding group keeps only the tail `K` blocks, so column `c` is at absolute position `base + c` with `base != 0`. Without the offset, `past_indices` (max ≈ K) never exceeds `context_lens` (≈ 100k) and `past_mask` becomes all-False — the layer silently attends to nothing. This only triggers when `block_list is not None` (prefix caching / later chunks), which your target run uses.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_tests/test_sliding_window_kv_spec.py`:

```python
import torch


def _build_past_mask(context_len, seq_len, window, K, block_size, base):
    """Reference re-implementation of the fixed mask (single request)."""
    device = "cpu"
    ctx = torch.tensor([context_len])
    invalid = ctx - window + torch.arange(seq_len) - 1  # [seq_len]
    past_indices = base + torch.arange(K)               # [K] absolute
    pm = ((past_indices.unsqueeze(0) > invalid.unsqueeze(-1)) &
          (past_indices.unsqueeze(0) < ctx.unsqueeze(-1)))  # [seq_len, K]
    return pm


def test_window_relative_mask_not_all_false_when_context_evicted():
    # Long context, small window: with base offset the mask must select
    # the in-window tail columns (non-empty). Without the offset it is all
    # False (the bug).
    context_len = 100_000
    window = 1024
    block_size = 128
    seq_len = 128
    K = math.ceil((window - 1 + seq_len) / block_size) + 1  # kept blocks
    K_cols = K * block_size
    base = max(0, math.ceil(context_len / block_size) * block_size - K_cols)
    pm = _build_past_mask(context_len, seq_len, window, K_cols, block_size, base)
    # At least the last query row must attend to some in-window context col.
    assert pm[-1].any(), "window-relative past_mask must be non-empty"
    # Sanity: the buggy zero-base version would be all-False here.
    pm_buggy = _build_past_mask(context_len, seq_len, window, K_cols,
                                block_size, base=0)
    assert not pm_buggy.any(), "zero-base reproduces the bug (all-False)"
```

- [ ] **Step 2: Run test to verify it fails**

This test encodes the reference formula; it should PASS immediately as a spec of intended behavior. To make it a true red-green for the production code, also add a test that calls the real method with a small fake metadata and asserts the mask is non-empty. If a full `attn_metadata` is too heavy to fake in a unit test, mark the production-call test `@pytest.mark.skip(reason="needs device metadata; covered by Task 6 e2e")` and rely on the reference test + Task 6.

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -k window_relative -v`
Expected: PASS for the reference test (documents the target formula).

- [ ] **Step 3: Write minimal implementation**

Replace lines 7077-7085 in `_set_attn_bias_for_sliding_window`:

```python
            block_list = attn_metadata.block_list
            max_context_len = (block_list.size(-1) // batch_size if block_list is not None else 0)
            block_size = getattr(prefill_metadata, "block_size", self.block_size)
            max_context_len = max_context_len * block_size

            if getattr(self, "use_sliding_window_kv", False):
                # Option A: only the tail `max_context_len` window columns are
                # present. Map column c -> absolute position base + c, where
                # base aligns the kept window to the end of the real context.
                aligned_ctx = ((context_lens_t + block_size - 1) // block_size) * block_size
                base = torch.clamp(aligned_ctx - max_context_len, min=0)  # [batch]
                past_indices = (base.unsqueeze(-1)
                                + torch.arange(max_context_len, device=device))  # [batch, K]
                invalid_lens_t = context_lens_t.unsqueeze(-1) - window_size \
                    + torch.arange(seq_len, device=device).unsqueeze(0) - 1     # [batch, seq_len]
                past_mask = ((past_indices.unsqueeze(1) > invalid_lens_t.unsqueeze(-1))
                             & (past_indices.unsqueeze(1) < context_lens_t.view(-1, 1, 1))
                             ).unsqueeze(1)  # [batch, 1, seq_len, K]
            else:
                invalid_lens_t = context_lens_t - window_size + torch.arange(seq_len, device=device) - 1
                past_indices = torch.arange(max_context_len, device=device)
                past_mask = ((past_indices.unsqueeze(0) > invalid_lens_t.unsqueeze(-1)) &
                             (past_indices.unsqueeze(0) < context_lens_t.unsqueeze(-1).unsqueeze(0))).unsqueeze(1)
```

Keep the `causal_mask` block (lines 7088-7090) and the `concat`/`where` (7098-7100) unchanged. Verify the broadcast shapes of `past_mask` and `causal_mask` still concatenate on `dim=-1` (both `[batch, 1, seq_len, *]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/litang/github/gemma4/vllm-gaudi && python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/litang/github/gemma4/vllm-gaudi
git add tests/unit_tests/test_sliding_window_kv_spec.py vllm_gaudi/v1/worker/hpu_model_runner.py
git commit -m "fix(hpu): window-relative prefill mask for sliding-window KV (Option A)"
```

---

### Task 6: End-to-end verification on Gaudi3

**Files:**
- Use: `test/run_enable_gemma4_482.sh` (existing launch), `test/run_gemma4_longbench.sh`, `test/run_gemma4_gsm8k.sh`
- No production code change; this task validates the feature and captures numbers.

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: a before/after memory + correctness report.

- [ ] **Step 1: Baseline (flag OFF) — confirm no regression**

```bash
cd /home/litang/github/gemma4/test
unset VLLM_HPU_SLIDING_WINDOW_KV   # default off
bash run_gemma4_gsm8k.sh 2>&1 | tee /tmp/gsm8k_flag_off.log
```
Expected: runs succeed; `get_kv_cache_spec` emits all `FullAttentionSpec`; log shows the same `Usable num_blocks` / `GPU KV cache size` as the pre-change baseline (`enable_gemma4_482.log`: `766,348 tokens`, concurrency `6.39x`).

- [ ] **Step 2: Feature ON — confirm SlidingWindowSpec + more KV capacity**

```bash
cd /home/litang/github/gemma4/test
export VLLM_HPU_SLIDING_WINDOW_KV=1
bash run_enable_gemma4_482.sh 2>&1 | tee /tmp/swa_flag_on.log
grep -E "Sliding-window KV cache .* ENABLED|GPU KV cache size|Maximum concurrency|Usable num_blocks|max model len" /tmp/swa_flag_on.log
```
Expected: log shows "Sliding-window KV cache (Option A) ENABLED"; `GPU KV cache size` (tokens) substantially **higher** than baseline; `Maximum concurrency` higher than 6.39x; `max model len` reaches the requested 128k (131072) instead of auto-capping to 119918.

- [ ] **Step 3: Correctness — accuracy parity**

```bash
cd /home/litang/github/gemma4/test
export VLLM_HPU_SLIDING_WINDOW_KV=1
bash run_gemma4_gsm8k.sh 2>&1 | tee /tmp/gsm8k_flag_on.log
# Compare accuracy to /tmp/gsm8k_flag_off.log (Step 1). Must match within noise.
```
Expected: GSM8K accuracy with flag ON matches flag OFF (sliding layers are mathematically identical up to the window; any divergence signals a mask bug from Task 5).

- [ ] **Step 4: Long-context correctness (exercises prefix caching + eviction)**

```bash
cd /home/litang/github/gemma4/test
export VLLM_HPU_SLIDING_WINDOW_KV=1
bash run_gemma4_longbench.sh 2>&1 | tee /tmp/longbench_flag_on.log
```
Expected: LongBench completes without the "too many values to unpack"-class errors, and scores match a flag-OFF LongBench run within noise. This is the key test: it drives multi-chunk prefill with `block_list is not None`, the exact path Task 5 fixes.

- [ ] **Step 5: Record results**

Write `/tmp/option_a_report.md` capturing: KV cache size (tokens) OFF vs ON, max concurrency OFF vs ON, whether 128k fits, GSM8K accuracy OFF vs ON, LongBench score OFF vs ON. Commit it to the repo docs if desired.

```bash
cd /home/litang/github/gemma4/vllm-gaudi
git add docs/superpowers/plans/2026-07-13-gemma4-sliding-window-kv-spec.md
git commit -m "docs(hpu): Option A sliding-window KV plan + e2e results"
```

---

## Test Plan Summary

| Layer | Test | What it proves |
|---|---|---|
| Flag gating | `test_sliding_window_kv_spec.py::test_flag_*` (Task 1) | Off by default; on only for interleaved-SWA + hybrid-enabled |
| Spec emission | `test_sliding_layer_emits_sliding_spec_when_enabled`, `test_all_full_when_disabled` (Task 2) | Sliding layers → `SlidingWindowSpec`; OFF path unchanged |
| Group resolution | `test_group_id_resolution` (Task 3) | Full vs sliding group ids resolved by spec type |
| KV allocation | Task 3b (startup, on device) | `SlidingWindowSpec` layers get KV tensors; no startup crash |
| Bucketing | `test_decode_bucket_hybrid.py::test_sliding_window_bucket_bounded_by_window` (Task 4) | Sliding decode bucket bounded by window, not full pool |
| Mask correctness | `test_window_relative_mask_not_all_false_when_context_evicted` (Task 5) | Window-relative `past_mask` non-empty; zero-base reproduces the bug |
| End-to-end | GSM8K + LongBench, OFF vs ON (Task 6) | Memory win + accuracy parity on device |

**Unit tests run on CPU** (no HPU needed) via `python -m pytest tests/unit_tests/test_sliding_window_kv_spec.py tests/unit_tests/test_decode_bucket_hybrid.py -v`.
**Task 6 requires Gaudi3** and the target checkpoint.

## Open Risks (validate during execution, not blockers)

1. **`MultiGroupBlockTable` with two *attention* groups** has only been exercised for attn+mamba on HPU. Task 3/6 confirm the `append_row`/`compute_slot_mapping` fan-out behaves for full+sliding. If block allocation for the second attention group misbehaves, that surfaces in Task 6 Step 4 (LongBench) as wrong output or an allocation error.
2. **`_resolve_all_blocks` / contiguous-PA `_PAD_BLOCK_ID`** may assume a single block-id space. If the sliding group's ids collide with the full group's pad sentinel, Task 6 shows wrong KV gather. Mitigation: per-group pad id (extend Task 4 if observed).
3. **(Resolved → Task 3b)** `initialize_kv_cache` raises `ValueError` on `SlidingWindowSpec` in the non-hybrid path; Task 3b fixes the `isinstance` guard. Remaining sub-risk: the two *hybrid/mamba-path* allocation branches (lines 6158 and 6264) also gate on `FullAttentionSpec` — only relevant if a future model interleaves sliding-window **and** mamba; out of scope for gemma4.
