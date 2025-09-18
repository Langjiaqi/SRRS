#!/bin/bash
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

#nccl
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO


# 强制启用离线模式，禁止任何网络连接
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 设置合适的缓存目录
export TRANSFORMERS_CACHE_DIR=/tmp/transformers_cache
export HF_DATASETS_CACHE=/tmp/datasets_cache
export HUGGINGFACE_HUB_CACHE=/tmp/hub_cache
export HF_HOME=/tmp/hf_home

# 创建缓存目录
mkdir -p /tmp/transformers_cache
mkdir -p /tmp/datasets_cache
mkdir -p /tmp/hub_cache
mkdir -p /tmp/hf_home

# 打印调试信息
echo "=== 环境变量设置 ==="
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "HF_DATASETS_OFFLINE: $HF_DATASETS_OFFLINE"
echo "模型路径: /llms/DeepSeek-R1-Distill-Llama-8B"
echo "==================="

forget_losses=(
    FINETUNE
)


task_list=(1)

learning_rates=(
    1e-5
)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
#model_path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
model_path=/llms/DeepSeek-R1-Distill-Llama-8B
mask=true

use_LoRA=false
save_root=results/rtofu

forget_coeff=1.0
regularization_coeff=1.0

fix_ref_model=false
save_checkpoint=false

save_steps=last
eval_steps=(last)

num_epochss=(10)
split=forget100
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
        done
    done
done
