for task_id in "${task_list[@]}"; do
    for forget_loss in "${forget_losses[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for macro_epoch in "${macro_epochs[@]}"; do
                COMMON="use_LoRA=$use_LoRA lr=$lr split=$split num_epochs=$num_epochs \
                    save_root=$save_root model_path=$model_path"
                CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 --master_port=$MASTER_PORT \
                    grpo.py \
                    --config-name=grpo.yaml \
                    task_id=$task_id \
                    $COMMON
            done
        done
    done
done





for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nproc_per_node=3 --master_port=$MASTER_PORT \
                    sftwithgrpo.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
        done
    done
done

