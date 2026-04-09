#!/bin/bash
# 诊断 Qwen3.5-397B warmup 问题

echo "=========================================="
echo "Qwen3.5-397B Warmup 诊断"
echo "=========================================="

# 检查环境变量
echo ""
echo "1. 检查环境变量设置："
echo "   VLLM_SPLIT_MOE_COMPILATION=$VLLM_SPLIT_MOE_COMPILATION"
echo "   VLLM_MOE_GRAPH_BREAK=$VLLM_MOE_GRAPH_BREAK"
echo "   VLLM_EXPONENTIAL_BUCKETING=$VLLM_EXPONENTIAL_BUCKETING"
echo "   VLLM_CONTIGUOUS_PA=$VLLM_CONTIGUOUS_PA"
echo "   VLLM_DEVELOPER_MODE=$VLLM_DEVELOPER_MODE"
echo "   ENABLE_EXPERIMENTAL_FLAGS=$ENABLE_EXPERIMENTAL_FLAGS"
echo "   ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=$ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES"

# 检查 HPU 状态
echo ""
echo "2. 检查 HPU 状态："
hlsmi --list 2>/dev/null || echo "   hlsmi 不可用"

# 检查日志中是否启用了 split compilation
echo ""
echo "3. 检查日志文件（如果有）："
if [ -f "$1" ]; then
    echo "   搜索 Split-compiling MoE 日志..."
    grep -c "Split-compiling MoE" "$1" 2>/dev/null && echo "   ✓ Split compilation 已启用" || echo "   ✗ 未找到 Split compilation 日志"
    
    echo "   搜索 warmup 进度..."
    grep "Prompt warmup processing" "$1" 2>/dev/null | tail -5
else
    echo "   未提供日志文件，请重启并添加日志重定向"
fi

echo ""
echo "=========================================="
echo "建议："
echo "1. 如果 Split compilation 未启用，检查环境变量是否正确传递"
echo "2. 如果 warmup 超过 30 分钟无进展，考虑重启并减少 buckets"
echo "3. 使用 --log-level debug 查看详细编译日志"
echo "=========================================="
