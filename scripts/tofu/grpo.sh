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


task_list=(1 2 3 4 5 6 7 8 9 10)

learning_rates=(

    1e-5
    5e-6
    2e-6

)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
model_path=/ljq/rtofu/results/rtofu/llama3-8b/forget100/FINETUNE/seed_1001/epoch10_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last

use_LoRA=false
save_root=results/rtofu_grpo_test_new

num_epochss=(3)
split=forget01

for num_epochs in "${num_epochss[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for task_id in "${task_list[@]}"; do
            COMMON="use_LoRA=$use_LoRA lr=$lr split=$split num_epochs=$num_epochs \
                save_root=$save_root model_path=$model_path"
            CUDA_VISIBLE_DEVICES=1,0,3 timeout 7200s torchrun --nproc_per_node=3 --master_port=$MASTER_PORT \
                grpo.py \
                --config-name=grpo.yaml \
                task_id=$task_id \
                $COMMON
        done
    done
done
