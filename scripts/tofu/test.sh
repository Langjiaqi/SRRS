#!/bin/bash
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置CUDA 11.8路径到临时变量
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_ROOT=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 跳过DeepSpeed的CUDA版本检查
export DS_SKIP_CUDA_CHECK=1

#export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

#nccl
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO


# 强制启用离线模式，禁止任何网络连接
#export TRANSFORMERS_OFFLINE=1
#export HF_DATASETS_OFFLINE=1
#export HF_HUB_OFFLINE=1

# 设置合适的缓存目录
export TRANSFORMERS_CACHE_DIR=/llms
export HF_DATASETS_CACHE=/llms/datasets_cache
export HUGGINGFACE_HUB_CACHE=/llms/hub_cache
export HF_HOME=/llms/hf_home

# 创建缓存目录
mkdir -p /llms
mkdir -p /llms/datasets_cache
mkdir -p /llms/hub_cache
mkdir -p /llms/hf_home

# 打印调试信息
echo "=== 环境变量设置 ==="
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "HF_DATASETS_OFFLINE: $HF_DATASETS_OFFLINE"
echo "模型路径: /llms/DeepSeek-R1-Distill-Llama-8B"
echo "==================="


forget_losses=(
################ GA
    GA1             ##CoT+Answer
    GA2             ##Answer-only
  
################ GD
    GA1+GD          ##CoT+Answer                 
    GA2+GD          ##Answer-only
   
################ KL
    GA1+KL          ##CoT+Answer    
    GA2+KL          ##Answer-only
   
################ PO
    IDK1+GD         ##Direct IDK
    IDK2+GD         ##Answer IDK
   
)


task_list=(1)

learning_rates=(
    1e-5
   
)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
model_path=/ljq/rtofu/results/rtofu/llama3-8b/forget100/FINETUNE/seed_1001/epoch10_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last
fix_ref_model=false  # 添加这一行

use_LoRA=false
save_root=results/rtofu

forget_coeff=1.0
regularization_coeff=1.0
mask=true
save_checkpoint=false

save_steps=last
eval_steps=(last)

num_epochss=(3)
split=forget10
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
            
        done
    done
done
