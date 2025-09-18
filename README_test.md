# 模型多输出测试脚本使用说明

## 概述
这个脚本用于测试指定路径的模型，支持多输出生成模式。基于 `eval.py` 修改，增加了多样性采样功能。

## 文件说明

- `testmodel.py` - 主测试脚本
- `test_config.py` - 配置文件，可以修改测试参数
- `run_test_multi.py` - 简化的启动脚本
- `README_test.md` - 这个说明文档

## 快速开始

### 1. 直接运行（使用默认配置）
```bash
cd /ljq/rtofu
python testmodel.py
```

### 2. 使用简化启动脚本
```bash
cd /ljq/rtofu
python run_test_multi.py
```

### 3. 自定义配置运行
1. 编辑 `test_config.py` 文件中的参数
2. 运行测试脚本：
```bash
python testmodel.py
```

## 配置参数说明

### 路径配置
- `MODEL_PATH` - 模型检查点路径
- `DATA_PATH` - 测试数据路径
- `OUTPUT_DIR` - 结果输出目录

### 多输出生成配置
- `NUM_ROLLOUTS` - 每个问题生成几个回答（默认：5）
- `TEMPERATURE` - 采样温度，越高越随机（默认：0.8）
- `TOP_P` - top_p采样参数（默认：0.9）
- `TOP_K` - top_k采样参数（默认：50）
- `MAX_NEW_TOKENS` - 最大生成token数（默认：256）
- `DO_SAMPLE` - 是否启用采样（默认：True）

### 测试配置
- `BATCH_SIZE` - 批次大小（默认：2）
- `DS_SIZE` - 从每个数据文件中取多少条数据（默认：50）
- `VERBOSE` - 是否启用详细输出（默认：True）

## 输出结果

脚本会在指定的输出目录生成以下文件：

1. `test_results_multi_output.json` - 详细结果，包含每个问题的所有生成回答
2. `test_summary_multi_output.csv` - 汇总结果，包含评估指标

## 评估指标

- `rouge1_recall_best` - 每个问题选择最佳回答的ROUGE-1召回率
- `rougeL_recall_best` - 每个问题选择最佳回答的ROUGE-L召回率
- `rouge1_recall_avg` - 所有回答的平均ROUGE-1召回率
- `rougeL_recall_avg` - 所有回答的平均ROUGE-L召回率

## 支持的数据文件

脚本会自动寻找以下数据文件：
- `forget_perturbed.json`
- `retain_perturbed.json`
- `real_authors_perturbed.json`
- `world_facts_perturbed.json`

如果找不到这些文件，会使用示例数据进行测试。

## 注意事项

1. 确保有足够的GPU内存来运行模型
2. 多输出生成会显著增加推理时间
3. 可以通过调整 `BATCH_SIZE` 和 `NUM_ROLLOUTS` 来平衡速度和内存使用
4. 如果遇到内存不足，可以减少 `NUM_ROLLOUTS` 或 `BATCH_SIZE`

## 故障排除

### 内存不足
- 减少 `NUM_ROLLOUTS`
- 减少 `BATCH_SIZE`
- 减少 `MAX_NEW_TOKENS`

### 模型加载失败
- 检查 `MODEL_PATH` 是否正确
- 确保模型文件完整且可访问

### 数据加载失败
- 检查 `DATA_PATH` 是否正确
- 确认数据文件格式正确 