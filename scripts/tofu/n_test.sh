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

# 多回复生成参数
export NUM_ROLLOUTS=5        # 每个prompt生成的回复数量
export TEMPERATURE=0.3      # 采样温度
export USE_VLLM=true        # 是否使用vLLM

# 打印调试信息
echo "=== 环境变量设置 ==="
echo "NUM_ROLLOUTS: $NUM_ROLLOUTS"
echo "TEMPERATURE: $TEMPERATURE"
echo "USE_VLLM: $USE_VLLM"
echo "模型路径: /llms/DeepSeek-R1-Distill-Llama-8B"
echo "==================="

forget_losses=(
################ GA
   IDK3+GD    
    
)

task_list=(1)

learning_rates=(
    1e-5
   
)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
model_path=/ljq/rtofu/results/rtofu/llama3-8b/forget100/FINETUNE/seed_1001/epoch10_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last
mask=true
fix_ref_model=false

use_LoRA=false
save_root=results/rtofu

forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false

save_steps=last
eval_steps=(last)

num_epochss=(3)
split=forget01

for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                
                # 训练部分（如果需要的话）
                # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
                #     forget.py \
                #     --config-name=tofu.yaml \
                #     task_id=$task_id \
                #     save_steps=$save_steps \
                #     $COMMON
            done
            
            # 使用n_eval.py进行多回复生成评估
            for step in "${eval_steps[@]}"; do
                echo "=== 开始多回复生成评估 ==="
                echo "评估步骤: $step"
                echo "Rollouts: $NUM_ROLLOUTS"
                echo "Temperature: $TEMPERATURE"
                echo "使用vLLM: $USE_VLLM"
                
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT \
                    n_eval.py \
                    --config-name=n_eval.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    generation.num_rollouts=$NUM_ROLLOUTS \
                    generation.temperature=$TEMPERATURE \
                    generation.use_vllm=$USE_VLLM \
                    $COMMON
                    
                echo "=== 评估完成 ==="
            done
        done
    done
done

echo "=== 所有评估任务完成 ===" 