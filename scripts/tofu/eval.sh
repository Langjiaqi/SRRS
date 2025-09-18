#!/bin/bash
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

forget_losses=(
################ GA
    GA3        ##CoT+Answer
  
)


task_list=(1)

learning_rates=(
    1e-5
  
)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
model_path=/ljq/rtofu/results/rtofu/llama3-8b/forget100/FINETUNE/seed_1001/epoch10_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last

use_LoRA=false
save_root=results/rtofu
save_dir=$model_path
forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false
fix_ref_model=false

save_steps=last
eval_steps=(last)
eval_unlearn_step=0
num_epochss=(3)
split=forget01
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path eval_unlearn_step=$eval_unlearn_step"
                
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
        
        done
    done
done
