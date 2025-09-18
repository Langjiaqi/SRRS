#!/bin/bash
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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

echo "==================="


forget_losses=(
################ GA
    GA1             ##CoT+Answer
    
)


task_list=(1)

learning_rates=(
    1e-5
   
)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
model_path=/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch3_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last
mask=true
fix_ref_model=false  # 添加这一行

use_LoRA=false
save_root=results/multi_rtofu

forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false

save_steps=last
eval_steps=(last)

num_epochss=(3)
split=forget01

# 定义超参数数组
num_rollouts_list=(128)
temperature_list=(1)
use_vllm_list=(false)

# 嵌套循环进行超参数扫描


echo "=== 所有超参数扫描完成 ===" 


for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                for num_rollouts in "${num_rollouts_list[@]}"; do
                    for temperature in "${temperature_list[@]}"; do
                        for use_vllm in "${use_vllm_list[@]}"; do
                            COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                            mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                            echo "=== 运行实验 ==="
                            echo "num_rollouts: $num_rollouts"
                            echo "temperature: $temperature"
                            echo "use_vllm: $use_vllm"
                            echo "==============="

                            # 运行单个实验
                            CUDA_VISIBLE_DEVICES=0 python -u \
                                n_eval.py \
                                --config-name=n_eval.yaml \
                                eval.num_rollouts=$num_rollouts \
                                eval.temperature=$temperature \
                                eval.use_vllm=$use_vllm \
                                $COMMON

                            echo "=== 实验完成 ==="
                            echo ""

                            # 重新生成端口避免冲突
                            MASTER_PORT=$((RANDOM % 50001 + 10000))
                        done
                    done
                done
            done
        done
    done
done