#!/bin/bash
# 随机端口，避免冲突
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL 设置 - 禁用多卡通信，避免混卡卡死
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=INFO

# 缓存目录设置
export TRANSFORMERS_CACHE_DIR=/llms
export HF_DATASETS_CACHE=/llms/datasets_cache
export HUGGINGFACE_HUB_CACHE=/llms/hub_cache
export HF_HOME=/llms/hf_home

# 创建缓存目录
mkdir -p /llms /llms/datasets_cache /llms/hub_cache /llms/hf_home

echo "==================="

forget_losses=(
    GA1   ## CoT+Answer
)

task_list=(1)
learning_rates=(1e-5)
model_path=/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch3_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last
mask=true
fix_ref_model=false
use_LoRA=false
save_root=results/multi_rtofu
forget_coeff=1.0
regularization_coeff=1.0
save_checkpoint=false
save_steps=last
eval_steps=(last)
num_epochss=(3)
split=forget01

# 超参数列表
num_rollouts_list=(3 5 10)
temperature_list=(0.1 0.3 0.7)
use_vllm_list=(true)

echo "=== 所有超参数扫描开始 ===" 

# 循环执行实验
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

                            # 选择单卡运行（这里固定卡 1，可以改成循环多卡）
                            export CUDA_VISIBLE_DEVICES=1

                            torchrun \
                                --nproc_per_node=1 \
                                --master_port=$MASTER_PORT \
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
