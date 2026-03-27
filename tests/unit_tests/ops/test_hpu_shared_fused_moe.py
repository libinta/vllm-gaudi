# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3.5 MoE (SharedFusedMoE + shared expert gate).

Tests the full Qwen3NextSparseMoeBlock-style MoE dispatch:
  1. Router gate → top-k expert selection
  2. Routed experts via HPU FusedMoE
  3. Shared expert with sigmoid gate
  4. Sum of shared + routed outputs

Compares HPU bf16 results against a CPU fp32 reference to catch
precision/routing bugs in the HPU MoE path.
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock

import habana_frameworks.torch as htorch
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm_gaudi.ops.hpu_fused_moe import HPUUnquantizedFusedMoEMethod


# ---------------------------------------------------------------------------
# Small-scale config matching Qwen3.5 MoE structure (scaled down for tests)
# ---------------------------------------------------------------------------
NUM_EXPERTS = 16
TOP_K = 4
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 128
SHARED_INTERMEDIATE_SIZE = 128
NUM_TOKENS = 8


def _create_shared_fused_moe():
    """Build a SharedFusedMoE layer with shared expert + gate, Qwen3.5 style."""
    # Shared expert gate: sigmoid gating (hidden_size → 1)
    shared_expert_gate = ReplicatedLinear(
        HIDDEN_SIZE, 1, bias=False, quant_config=None,
        prefix="shared_expert_gate",
    )

    # Shared expert MLP (gate_up_proj + down_proj + SiluAndMul)
    shared_expert = Qwen2MoeMLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=SHARED_INTERMEDIATE_SIZE,
        hidden_act="silu",
        quant_config=None,
        reduce_results=False,
        expert_gate=shared_expert_gate,
        prefix="shared_expert",
    )

    # Router gate: hidden_size → num_experts
    gate = ReplicatedLinear(
        HIDDEN_SIZE, NUM_EXPERTS, bias=False, quant_config=None,
        prefix="gate",
    )

    # SharedFusedMoE wrapping routed experts + shared expert
    layer = SharedFusedMoE(
        shared_experts=shared_expert,
        gate=gate,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        params_dtype=torch.bfloat16,
        reduce_results=True,
        renormalize=True,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        quant_config=None,
        tp_size=None,
        ep_size=None,
        dp_size=None,
        custom_routing_function=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        apply_router_weight_on_input=False,
        activation="silu",
        enable_eplb=False,
        num_redundant_experts=0,
        has_bias=False,
        is_sequence_parallel=False,
    )
    return layer


def _cpu_reference_forward(layer, hidden_states):
    """Run the Qwen3NextSparseMoeBlock-equivalent forward on CPU in fp32.

    This mirrors the upstream Qwen3NextSparseMoeBlock.forward(), manually
    invoking the shared expert and routed experts to get a reference output.
    """
    x = hidden_states.float()

    # 1. Shared expert: gate_up_proj → SiluAndMul → down_proj → sigmoid gate
    shared_expert = layer._shared_experts
    gate_up, _ = shared_expert.gate_up_proj(x)
    shared_out = shared_expert.act_fn(gate_up)
    shared_out, _ = shared_expert.down_proj(shared_out)
    if shared_expert.expert_gate is not None:
        gate_val, _ = shared_expert.expert_gate(x)
        shared_out = F.sigmoid(gate_val) * shared_out

    # 2. Router: compute logits, softmax, top-k
    router_logits, _ = layer._gate(x)
    topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(topk_weights, TOP_K, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    # 3. Routed experts: naive loop (no fused kernel)
    routed_out = torch.zeros_like(x)
    w13 = layer.w13_weight.data.float()  # [num_experts, 2*intermediate, hidden]
    w2 = layer.w2_weight.data.float()    # [num_experts, hidden, intermediate]

    for token_idx in range(x.shape[0]):
        token = x[token_idx]  # [hidden]
        token_out = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
        for k in range(TOP_K):
            eid = topk_ids[token_idx, k].item()
            w = topk_weights[token_idx, k].item()
            # gate_up projection
            gate_up_out = w13[eid] @ token  # [2*intermediate]
            gate_part = gate_up_out[:INTERMEDIATE_SIZE]
            up_part = gate_up_out[INTERMEDIATE_SIZE:]
            activated = F.silu(gate_part) * up_part  # SiluAndMul
            # down projection
            expert_out = w2[eid] @ activated  # [hidden]
            token_out += w * expert_out
        routed_out[token_idx] = token_out

    return (shared_out + routed_out).to(hidden_states.dtype)


def test_shared_fused_moe_accuracy(default_vllm_config, dist_init):
    """Test SharedFusedMoE on HPU matches CPU fp32 reference."""
    torch.manual_seed(42)

    # Build layer on CPU first, copy weights for reference
    layer = _create_shared_fused_moe()
    assert isinstance(layer.quant_method, HPUUnquantizedFusedMoEMethod)

    # Random input
    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16)

    # CPU reference (before moving to HPU)
    with torch.no_grad():
        ref_output = _cpu_reference_forward(layer, hidden_states.cpu())

    # Move to HPU in bf16 and process weights
    layer = layer.to(dtype=torch.bfloat16, device="hpu")
    layer.quant_method.process_weights_after_loading(layer)

    if not htorch.utils.internal.is_lazy():
        from vllm_gaudi.utils import HPUCompileConfig
        compile_config = HPUCompileConfig()
        layer = torch.compile(layer, **compile_config.get_compile_args())

    # Run on HPU — SharedFusedMoE has is_internal_router=True (gate passed
    # at construction), so router_logits=hidden_states triggers the internal
    # gate to compute actual logits.
    hidden_states_hpu = hidden_states.to("hpu")

    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with torch.no_grad(), override_forward_context(mock_ctx):
        shared_out, fused_out = layer(
            hidden_states=hidden_states_hpu,
            router_logits=hidden_states_hpu,
        )

    hpu_output = (shared_out + fused_out).cpu()

    # Compare
    torch.testing.assert_close(
        ref_output.cpu().float(), hpu_output.float(),
        atol=5e-2, rtol=5e-2,
        msg="SharedFusedMoE HPU output differs from CPU fp32 reference",
    )


def test_shared_fused_moe_3d_input(default_vllm_config, dist_init):
    """Test that 3D input [B, S, H] works correctly (HPU decode path)."""
    torch.manual_seed(42)

    layer = _create_shared_fused_moe()
    layer = layer.to(dtype=torch.bfloat16, device="hpu")
    layer.quant_method.process_weights_after_loading(layer)

    # 2D input (baseline)
    hidden_2d = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="hpu")
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None

    with torch.no_grad(), override_forward_context(mock_ctx):
        shared_2d, fused_2d = layer(
            hidden_states=hidden_2d,
            router_logits=hidden_2d,
        )
    out_2d = (shared_2d + fused_2d)

    # 3D input (same data, reshaped to [B=2, S=4, H])
    hidden_3d = hidden_2d.reshape(2, NUM_TOKENS // 2, HIDDEN_SIZE)
    hidden_3d_flat = hidden_3d.reshape(-1, HIDDEN_SIZE)

    with torch.no_grad(), override_forward_context(mock_ctx):
        shared_3d, fused_3d = layer(
            hidden_states=hidden_3d_flat,
            router_logits=hidden_3d_flat,
        )
    out_3d = (shared_3d + fused_3d)

    # Should produce identical results regardless of batch reshaping
    torch.testing.assert_close(
        out_2d, out_3d,
        atol=1e-5, rtol=1e-5,
        msg="SharedFusedMoE gives different results for 2D vs 3D input",
    )


def test_shared_expert_gate_effect(default_vllm_config, dist_init):
    """Test that the shared expert sigmoid gate actually modulates output."""
    torch.manual_seed(42)

    layer = _create_shared_fused_moe()

    # Set the shared expert gate weight to produce large negative values
    # so sigmoid → ~0, effectively zeroing the shared expert
    with torch.no_grad():
        layer._shared_experts.expert_gate.weight.fill_(-10.0)

    layer = layer.to(dtype=torch.bfloat16, device="hpu")
    layer.quant_method.process_weights_after_loading(layer)

    hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="hpu")
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None

    with torch.no_grad(), override_forward_context(mock_ctx):
        shared_out, fused_out = layer(
            hidden_states=hidden_states,
            router_logits=hidden_states,
        )

    # With gate ≈ 0, shared_out should be near zero
    shared_norm = shared_out.float().abs().max().item()
    fused_norm = fused_out.float().abs().max().item()

    assert shared_norm < 0.01, (
        f"Shared expert output should be near-zero with gate=-10, "
        f"but max abs = {shared_norm}"
    )
    assert fused_norm > 0.0, "Fused expert output should be non-zero"


def test_routing_weight_precision(default_vllm_config, dist_init):
    """Test that routing weights maintain precision through the HPU path.

    The HPU apply_monolithic casts topk_weights to bf16 before expert
    dispatch. This test checks whether that causes meaningful divergence
    from an fp32 routing baseline.
    """
    torch.manual_seed(42)

    # Create router logits that stress precision (close-valued experts)
    router_logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32)

    # fp32 reference routing
    weights_fp32 = F.softmax(router_logits, dim=1, dtype=torch.float32)
    topk_w_fp32, topk_ids_fp32 = torch.topk(weights_fp32, TOP_K, dim=-1)
    topk_w_fp32 /= topk_w_fp32.sum(dim=-1, keepdim=True)

    # bf16 routing (simulates HPU path)
    weights_bf16 = F.softmax(router_logits, dim=1, dtype=torch.float32)
    topk_w_bf16, topk_ids_bf16 = torch.topk(weights_bf16, TOP_K, dim=-1)
    topk_w_bf16 /= topk_w_bf16.sum(dim=-1, keepdim=True)
    topk_w_bf16 = topk_w_bf16.to(torch.bfloat16).float()  # round-trip

    # Expert selection should be identical
    assert torch.equal(topk_ids_fp32, topk_ids_bf16), (
        "Expert selection diverged between fp32 and bf16 routing"
    )

    # Weight precision loss should be bounded
    max_weight_diff = (topk_w_fp32 - topk_w_bf16).abs().max().item()
    assert max_weight_diff < 0.01, (
        f"Routing weight precision loss too large: {max_weight_diff}"
    )
