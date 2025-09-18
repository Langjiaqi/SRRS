# GRPO训练TensorBoard监控指南

## 概述

已为GRPO训练脚本添加了TensorBoard记录功能，可以实时监控训练过程中的各种指标。

## 记录的指标

### 基础训练指标
- **learning_rate**: 学习率变化
- **loss/total_loss**: 总损失值

### GRPO特有指标
- **grpo/reward**: 平均奖励值
- **grpo/kl_divergence**: KL散度（与参考模型的差异）

### 分别的奖励函数
- **rewards/rouge_recall_forget**: Rouge recall遗忘奖励
- **rewards/cosine_similarity_forget**: 余弦相似度遗忘奖励  
- **rewards/quality**: 回答质量奖励

### 优化指标
- **optimization/grad_norm**: 梯度范数

## 使用方法

### 1. 运行训练
正常运行GRPO训练脚本，TensorBoard日志会自动记录到：
```
{输出目录}/tensorboard_logs/
```

### 2. 查看TensorBoard

#### 方法一：使用便捷脚本
```bash
python start_tensorboard.py
```

这个脚本会：
- 自动查找所有可用的日志目录
- 如果找到多个，会让你选择要查看的目录
- 启动TensorBoard并显示访问地址

#### 方法二：手动启动
```bash
tensorboard --logdir path/to/tensorboard_logs --port 6006
```

### 3. 浏览器访问
在浏览器中打开：http://localhost:6006

## 脚本选项

`start_tensorboard.py` 支持以下选项：

```bash
python start_tensorboard.py --help
```

- `--logdir`: 指定特定的日志目录
- `--port`: 指定端口号（默认6006）
- `--host`: 指定主机地址（默认localhost）
- `--base_dir`: 指定搜索日志的基础目录

## 示例

### 查看特定实验的日志
```bash
python start_tensorboard.py --logdir grpo_seed_42/unlearn_times_1/tensorboard_logs
```

### 使用不同端口
```bash
python start_tensorboard.py --port 6007
```

### 允许外部访问
```bash
python start_tensorboard.py --host 0.0.0.0
```

## 训练监控建议

1. **损失趋势**: 观察total_loss是否稳定下降
2. **奖励平衡**: 检查各个奖励分量是否合理平衡
3. **KL散度**: 确保不会过度偏离参考模型
4. **梯度范数**: 监控是否出现梯度爆炸或消失
5. **学习率**: 确认学习率调度是否正常

## 注意事项

- TensorBoard日志每10个训练步记录一次
- 日志会自动保存，训练中断后可以继续查看
- 如果使用分布式训练，只有rank 0进程会记录日志
- 确保安装了tensorboard：`pip install tensorboard`

## 故障排除

### 找不到tensorboard命令
```bash
pip install tensorboard
```

### 端口被占用
使用不同端口：
```bash
python start_tensorboard.py --port 6007
```

### 权限问题
确保日志目录有读取权限：
```bash
chmod -R 755 path/to/tensorboard_logs
``` 