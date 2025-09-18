#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置文件
用户可以在这里修改测试参数
"""

# 模型和数据路径配置
MODEL_PATH = "/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch1_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last"
DATA_PATH = "/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch1_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/task_data"
OUTPUT_DIR = "/ljq/rtofu/test_results"

# 模型配置
MODEL_FAMILY = "llama3-8b"
USE_LORA = False
DEVICE_MAP = "auto"
CUDA_VISIBLE_DEVICES = "0"  # 设置可见的GPU，"0"表示只使用第一个GPU

# 多输出生成配置
NUM_ROLLOUTS = 5        # 每个问题生成几个回答
TEMPERATURE = 0.8       # 采样温度，越高越随机
TOP_P = 0.9            # top_p采样参数
TOP_K = 50             # top_k采样参数
MAX_NEW_TOKENS = 256   # 最大生成token数
DO_SAMPLE = True       # 是否启用采样

# 测试配置
BATCH_SIZE = 2         # 批次大小
DS_SIZE = 5           # 测试数据集大小（从每个数据文件中取多少条）

# 是否启用详细输出
VERBOSE = True 