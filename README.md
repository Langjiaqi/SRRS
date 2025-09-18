

## Requirements


- **Software**: Python 3.11, PyTorch, CUDA 11.8/12.4

## Installation

### 1. Create Environment

```shell
conda create -n rtofu python=3.11
conda activate rtofu
```

### 2. Install PyTorch and CUDA

```shell
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-12.4.1" cuda-toolkit
```

### 3. Install Dependencies

```shell
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```



## Core Functionality

### 1. GRPO Training
Machine unlearning training based on Generative Ranking Preference Optimization:

```shell
# Single-stage GRPO training
bash scripts/tofu/grpo.sh

# Or use Python directly
python grpo.py
```

### 2. SFT+GRPO Two-stage Training
First perform supervised fine-tuning, then execute GRPO optimization:

```shell
# Two-stage training
bash scripts/tofu/grposft.sh

# Or use Python directly
python grposft.py
```

### 3. Continual Unlearning Training
Support for progressive multi-task forgetting:

```shell
# Continual unlearning training
bash scripts/tofu/continual.sh
```

### 4. Model Evaluation

```shell
# Standard evaluation
bash scripts/tofu/eval.sh

# Task-specific evaluation
bash scripts/tofu/eval_for_specific.sh
```

### 5. Model Testing

```shell
# Multi-output generation testing
python testmodel.py

# Simplified testing
python run_test_multi.py
```


## Acknowledgments

This repository builds upon selected components of the codebase from:
- [A Closer Look at Machine Unlearning for Large Language Models](https://github.com/sail-sg/closer-look-LLM-unlearning)
- [R-TOFU original implementation](https://github.com/ssangyeon/R-TOFU)

We appreciate their outstanding work and contributions to the machine unlearning research community!

