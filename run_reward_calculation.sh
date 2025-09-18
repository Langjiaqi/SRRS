#!/bin/bash

# 奖励值计算脚本使用示例
# 修改以下参数以适配您的环境

# 模型路径（需要根据实际情况修改）
MODEL_PATH="/path/to/your/model"

# 数据集路径（需要根据实际情况修改）
DATA_PATH="/ljq/rtofu/data"

# 其他参数
SPLIT="train"
TASK_ID=1  # 或者使用-1加载所有任务
NUM_GENERATIONS=3
MAX_SAMPLES=10  # 限制样本数量以便快速测试，使用-1处理所有样本
OUTPUT_FILE="reward_results_task${TASK_ID}.json"

echo "开始计算奖励值..."
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "任务ID: $TASK_ID"
echo "生成数量: $NUM_GENERATIONS"
echo "最大样本数: $MAX_SAMPLES"

python calculate_reward.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --split "$SPLIT" \
    --task_id $TASK_ID \
    --num_generations $NUM_GENERATIONS \
    --max_samples $MAX_SAMPLES \
    --output_file "$OUTPUT_FILE"

echo "计算完成！结果保存在: $OUTPUT_FILE" 