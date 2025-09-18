#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型脚本 - 支持多输出生成
基于eval.py修改，使用指定的模型和数据路径
"""

import csv
import json
import os
import warnings
from typing import List, Dict, Any

# 导入配置并设置GPU
try:
    from test_config import CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
except ImportError:
    # 如果配置文件不存在，默认使用GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
from metrics.metrics import get_dataloader, get_eval_results
from utils.utils import get_model_identifiers_from_yaml

# 导入配置
try:
    from test_config import *
except ImportError:
    # 如果配置文件不存在，使用默认配置
    MODEL_PATH = "/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch1_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last"
    DATA_PATH = "/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch1_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/task_data"
    OUTPUT_DIR = "/ljq/rtofu/test_results"
    MODEL_FAMILY = "llama3-8b"
    USE_LORA = False
    DEVICE_MAP = "auto"
    CUDA_VISIBLE_DEVICES = "0"
    NUM_ROLLOUTS = 5
    TEMPERATURE = 0.8
    TOP_P = 0.9
    TOP_K = 50
    MAX_NEW_TOKENS = 256
    DO_SAMPLE = True
    BATCH_SIZE = 2
    DS_SIZE = 50
    VERBOSE = True

warnings.filterwarnings('ignore')

class ModelConfig:
    """模型配置类"""
    def __init__(self):
        self.model_family = MODEL_FAMILY
        self.model_path = MODEL_PATH
        self.data_path = DATA_PATH
        self.output_dir = OUTPUT_DIR
        self.use_LoRA = USE_LORA
        self.device_map = DEVICE_MAP
        
        # 多输出相关配置
        self.num_rollouts = NUM_ROLLOUTS  # 每个问题生成多少个回答
        self.temperature = TEMPERATURE   # 采样温度
        self.top_p = TOP_P              # top_p采样
        self.top_k = TOP_K              # top_k采样
        self.max_new_tokens = MAX_NEW_TOKENS
        self.do_sample = DO_SAMPLE
        
        # 评估配置
        self.batch_size = BATCH_SIZE    # 批次大小
        self.ds_size = DS_SIZE          # 测试数据集大小
        self.verbose = VERBOSE          # 是否详细输出

def load_model_and_tokenizer(config: ModelConfig):
    """加载模型和分词器"""
    print(f"正在加载模型: {config.model_path}")
    
    model_cfg = get_model_identifiers_from_yaml(config.model_family)
    model_id = model_cfg["hf_key"]
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # 设置为左填充，适用于生成任务
    
    # 加载模型配置
    model_config = AutoConfig.from_pretrained(model_id)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        model_config.rope_scaling.setdefault("type", "linear")
    
    # 加载模型
    if os.path.exists(config.model_path):
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            config=model_config,
            torch_dtype=torch.bfloat16,
            device_map=config.device_map,
            trust_remote_code=True
        )
    else:
        print(f"警告: 模型路径 {config.model_path} 不存在，使用基础模型")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=model_config,
            torch_dtype=torch.bfloat16,
            device_map=config.device_map,
            trust_remote_code=True
        )
    
    model.eval()
    print(f"模型加载完成，设备: {model.device}")
    return model, tokenizer

def generate_multiple_responses(
    model, 
    tokenizer, 
    input_texts: List[str], 
    test_data: List[Dict[str, Any]],
    config: ModelConfig
) -> List[List[str]]:
    """为每个输入文本生成多个回复"""
    all_responses = []
    
    for i, input_text in enumerate(input_texts):
        print(f"\n{'='*80}")
        print(f"正在处理第 {i+1}/{len(input_texts)} 个问题")
        print(f"{'='*80}")
        
        # 从原始数据中获取问题和标准答案
        original_question = test_data[i].get("question", "")
        ground_truth = test_data[i].get("answer", "")
        
        print(f"问题: {original_question}")
        print(f"标准答案: {ground_truth}")
        print(f"输入格式: {input_text}")
        print("-" * 80)
        
        # 编码输入
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024
        ).to(model.device)
        
        responses = []
        for rollout in range(config.num_rollouts):
            print(f"生成第 {rollout+1}/{config.num_rollouts} 个回复...")
            
            with torch.no_grad():
                # 设置不同的随机种子以获得多样性
                torch.manual_seed(42 + rollout)
                
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # 解码生成的文本
                generated_text = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # 后处理：提取答案部分
                if "</think>" in generated_text:
                    # 提取思考后的答案
                    answer = generated_text.split("</think>")[-1].strip()
                    if answer.startswith("\n\n"):
                        answer = answer[2:].strip()
                else:
                    answer = generated_text
                
                responses.append(answer)
                print(f"  回复 {rollout+1}: {answer}")
        
        print("-" * 80)
        print("本题所有回复:")
        for idx, resp in enumerate(responses, 1):
            print(f"  {idx}. {resp}")
        
        all_responses.append(responses)
    
    return all_responses

def prepare_test_data(config: ModelConfig) -> List[Dict[str, Any]]:
    """准备测试数据"""
    print(f"正在从 {config.data_path} 加载测试数据...")
    
    test_data = []
    
    # 尝试加载不同的数据文件
    possible_files = [
        "forget_perturbed.json",
        "retain_perturbed.json", 
        "real_authors_perturbed.json",
        "world_facts_perturbed.json"
    ]
    
    for filename in possible_files:
        filepath = os.path.join(config.data_path, filename)
        if os.path.exists(filepath):
            print(f"找到数据文件: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                # 尝试标准JSON格式
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # 如果失败，尝试JSONL格式（每行一个JSON对象）
                    f.seek(0)
                    data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                
            # 限制数据大小
            if isinstance(data, list):
                data = data[:config.ds_size//len(possible_files)]
                for item in data:
                    item['data_source'] = filename
                    test_data.append(item)
            
    if not test_data:
        # 如果没有找到标准文件，创建示例数据
        print("未找到标准数据文件，创建示例数据...")
        test_data = [
            {
                "question": "什么是人工智能？",
                "answer": "人工智能是计算机科学的一个分支...",
                "data_source": "sample"
            },
            {
                "question": "请解释深度学习的基本概念。",
                "answer": "深度学习是机器学习的一个子领域...",
                "data_source": "sample"
            }
        ]
    
    print(f"加载了 {len(test_data)} 条测试数据")
    return test_data

def format_input_text(item: Dict[str, Any], model_family: str) -> str:
    """格式化输入文本"""
    question = item.get("question", "")
    
    if model_family == "llama3-8b":
        # 使用适合llama3的格式
        formatted_text = f"<｜User｜>{question}<｜Assistant｜><think>\n"
    else:
        formatted_text = f"问题: {question}\n回答: "
    
    return formatted_text

def evaluate_responses(responses: List[List[str]], ground_truths: List[str]) -> Dict[str, float]:
    """评估生成的回复质量"""
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # 计算每个问题的最佳ROUGE分数（在多个回复中选择最高的）
    best_rouge1_scores = []
    best_rougeL_scores = []
    
    for response_list, gt in zip(responses, ground_truths):
        best_rouge1 = 0
        best_rougeL = 0
        
        for response in response_list:
            scores = scorer.score(gt, response)
            best_rouge1 = max(best_rouge1, scores['rouge1'].recall)
            best_rougeL = max(best_rougeL, scores['rougeL'].recall)
        
        best_rouge1_scores.append(best_rouge1)
        best_rougeL_scores.append(best_rougeL)
    
    return {
        "rouge1_recall_best": sum(best_rouge1_scores) / len(best_rouge1_scores),
        "rougeL_recall_best": sum(best_rougeL_scores) / len(best_rougeL_scores),
        "rouge1_recall_avg": sum([sum(scores)/len(scores) for scores in [[scorer.score(gt, resp)['rouge1'].recall for resp in resp_list] for resp_list, gt in zip(responses, ground_truths)]]) / len(responses),
        "rougeL_recall_avg": sum([sum(scores)/len(scores) for scores in [[scorer.score(gt, resp)['rougeL'].recall for resp in resp_list] for resp_list, gt in zip(responses, ground_truths)]]) / len(responses)
    }

def save_results(
    test_data: List[Dict[str, Any]], 
    responses: List[List[str]], 
    config: ModelConfig,
    evaluation_metrics: Dict[str, float]
):
    """保存测试结果"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 保存详细结果
    detailed_results = []
    for i, (item, response_list) in enumerate(zip(test_data, responses)):
        result = {
            "index": i,
            "question": item.get("question", ""),
            "ground_truth": item.get("answer", ""),
            "data_source": item.get("data_source", "unknown"),
            "generated_responses": response_list,
            "num_responses": len(response_list)
        }
        detailed_results.append(result)
    
    # 保存到JSON文件
    results_file = os.path.join(config.output_dir, "test_results_multi_output.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "model_path": config.model_path,
                "data_path": config.data_path,
                "num_rollouts": config.num_rollouts,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "max_new_tokens": config.max_new_tokens
            },
            "evaluation_metrics": evaluation_metrics,
            "detailed_results": detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    # 保存汇总结果到CSV
    summary_file = os.path.join(config.output_dir, "test_summary_multi_output.csv")
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "数值"])
        writer.writerow(["模型路径", config.model_path])
        writer.writerow(["数据路径", config.data_path])
        writer.writerow(["生成回复数", config.num_rollouts])
        writer.writerow(["温度", config.temperature])
        writer.writerow(["测试样本数", len(test_data)])
        for metric, value in evaluation_metrics.items():
            writer.writerow([metric, f"{value:.4f}"])
    
    print(f"结果已保存到:")
    print(f"  详细结果: {results_file}")
    print(f"  汇总结果: {summary_file}")

def main():
    """主函数"""
    print("=" * 60)
    print("模型多输出测试脚本")
    print("=" * 60)
    
    # 打印GPU信息
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: CUDA不可用")
    
    # 初始化配置
    config = ModelConfig()
    print(f"模型路径: {config.model_path}")
    print(f"数据路径: {config.data_path}")
    print(f"输出目录: {config.output_dir}")
    print(f"每个问题生成 {config.num_rollouts} 个回复")
    print("-" * 60)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 准备测试数据
    test_data = prepare_test_data(config)
    
    # 格式化输入文本
    input_texts = [format_input_text(item, config.model_family) for item in test_data]
    ground_truths = [item.get("answer", "") for item in test_data]
    
    print("-" * 60)
    print("开始生成多个回复...")
    
    # 生成多个回复
    responses = generate_multiple_responses(model, tokenizer, input_texts, test_data, config)
    
    print("-" * 60)
    print("开始评估结果...")
    
    # 评估结果
    evaluation_metrics = evaluate_responses(responses, ground_truths)
    
    # 打印评估结果
    print("\n评估结果:")
    for metric, value in evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存结果
    save_results(test_data, responses, config, evaluation_metrics)
    
    print("-" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
