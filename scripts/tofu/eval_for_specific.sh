#!/bin/bash
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 直接设置变量值，不需要数组
forget_loss=GA3
task_id=1
lr=1e-5
num_epochs=3
eval_unlearn_step=last
# 改地址和unlearn_times进行不同模型的测评
export TASK_LIST=$task_id
model_path=/ljq/rtofu/results/rtofu_grposft_last_sft50/llama3-8b/forget01/IDK3+GD/seed_1001/epoch1_5e-06_FixRefTrue_maskTrue_1.0_1.0/1-2-3-4-5-6-7-8-9-10/unlearn_times_10/checkpoint-last
save_root=results/rtofu
save_dir=$(dirname $model_path)
unlearn_times=10
forget_coeff=1.0
regularization_coeff=1.0
mask=true
save_checkpoint=false
fix_ref_model=false
split=forget01

# 构建通用参数
COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path eval_unlearn_step=0 unlearn_times=$unlearn_times \
    save_dir=$save_dir"

# 直接运行评估命令
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
    eval_for_specific.py \
    --config-name=tofu.yaml \
    task_id=$task_id \
    eval_unlearn_step=$eval_unlearn_step \
    $COMMON