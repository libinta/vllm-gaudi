#!/bin/bash
# Qwen3.5-397B 启动脚本
# 适用于 Intel Gaudi HPU

# ============================================
# 核心环境变量（必须设置）
# ============================================

# 1. 启用 Split MoE Compilation - 关键！
#    Qwen3.5-397B 有 256 个 experts，不启用会导致编译时间爆炸
export VLLM_SPLIT_MOE_COMPILATION=1

# 2. 启用 Exponential Bucketing - 减少 warmup 时间
#    从 ~29 分钟减少到 ~11 分钟
export VLLM_EXPONENTIAL_BUCKETING=1

# 3. 启用 Contiguous Paged Attention - 优化 KV cache 管理
export VLLM_CONTIGUOUS_PA=1

# 4. (可选) 设置 MoE 拆分阈值（默认 256）
#    Qwen3.5-397B 有 256 个 experts，默认阈值 > 256 不会自动启用
#    建议显式设置 VLLM_SPLIT_MOE_COMPILATION=1，或者降低阈值
# export VLLM_SPLIT_MOE_EXPERT_THRESHOLD=200

# 5. (重要) 减少 torch.compile recompilations - 推荐启用！
#    这个选项可以显著减少 Qwen3.5-397B 的 warmup 编译时间
export VLLM_MOE_GRAPH_BREAK=1

# 6. (重要) 减少 warmup buckets - 大幅减少编译时间！
#    默认 40 个 buckets 可能需要 30-60 分钟
#    减少到 10-15 个可以缩短到 10-20 分钟
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_MAX=4
export VLLM_PROMPT_CTX_BUCKET_MIN=2048
export VLLM_PROMPT_CTX_BUCKET_MAX=8192
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_MAX=8

# 7. (重要) Qwen3.5 GDN hybrid 模型必需！
#    这两个参数必须都设置为 true 才能改善 graph compilation 性能
export ENABLE_EXPERIMENTAL_FLAGS=true
export ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true

# ============================================
# 关于 WARNING 的说明
# ============================================
# 你可能会看到以下警告：
#   "Unknown vLLM environment variable detected: VLLM_SPLIT_MOE_COMPILATION"
#   "Unknown vLLM environment variable detected: VLLM_EXPONENTIAL_BUCKETING"
#   "Unknown vLLM environment variable detected: VLLM_CONTIGUOUS_PA"
#
# 这些警告可以忽略！
# 原因：这些变量是在 vllm-gaudi 插件中定义的，不是 vLLM 主程序的一部分。
# vLLM 主程序在启动时会检查环境变量列表，发现不认识这些变量就会发出警告。
# 只要 vllm-gaudi 插件正确加载，这些变量就会生效。

# ============================================
# 可选优化变量（根据需要设置）
# ============================================

# 减少 decode block bucket 数量（如果 warmup 仍然很慢）
# export VLLM_DECODE_BLOCK_BUCKET_MIN=6

# 启用开发者模式查看 warmup 进度
export VLLM_DEVELOPER_MODE=1

# 设置日志级别（DEBUG, INFO, WARNING, ERROR）
# 默认是 INFO，设置为 WARNING 可以减少 INFO 日志输出
# export VLLM_LOGGING_LEVEL=WARNING

# 启用 torch compile 调试信息
export TORCH_COMPILE_DEBUG=1

# ============================================
# 启动命令
# ============================================

# 根据你的硬件配置调整以下参数：
# --tensor-parallel-size: HPU 数量（通常是 8）
# --max-model-len: 最大上下文长度（397B 支持 131K）
# --block-size: 块大小（推荐 128）
# --gpu-memory-utilization: 内存利用率（推荐 0.95）
#
# 注意：不要使用 --enforce-eager！
#   Qwen3.5-397B 需要 torch.compile 来获得可接受的性能。
#   Split MoE Compilation 功能依赖 torch.compile。
#   首次编译需要时间，但编译后的性能远好于 eager 模式。

python -m vllm.entrypoints.api_server \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --block-size 128 \
    --gpu-memory-utilization 0.95 \
    $@
