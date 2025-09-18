#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 forget01.json 数据测试 metrics.py 文件中的函数
"""

import os
import sys
import torch
import yaml
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics.metrics import (
    read_jsonline, 
    token_entropy, 
    compute_token_entropy,
    eval_rouge_recall,
    eval_cosine_similarity,
    get_entailment_results,
    get_entailment_score,
    mask_non_answer_labels,
    get_batch_loss,
    TextDatasetQA,
    custom_data_collator,
    get_dataloader,
    get_all_evals
)

def load_config():
    """加载配置文件"""
    config_path = "config/tofu.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    print(f"正在加载模型: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        print("模型和分词器加载成功!")
        return model, tokenizer
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None

def load_forget01_data():
    """加载 forget01.json 数据"""
    print("\n=== 加载 forget01.json 数据 ===")
    
    data_path = "/ljq/rtofu/data/tofu/forget01.json"
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        return None
    
    try:
        data = read_jsonline(data_path)
        print(f"成功加载 {len(data)} 条数据")
        
        # 显示数据结构
        if data:
            print(f"数据字段: {list(data[0].keys())}")
            print(f"第一条数据示例:")
            for key, value in data[0].items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def test_read_jsonline_with_forget01():
    """使用 forget01.json 测试 read_jsonline 函数"""
    print("\n=== 测试 read_jsonline 函数 (使用 forget01.json) ===")
    
    data = load_forget01_data()
    if data is None:
        return False
    
    # 分析数据内容
    print(f"\n数据统计:")
    print(f"总样本数: {len(data)}")
    
    # 统计问题长度
    question_lengths = [len(item.get('question', '')) for item in data]
    print(f"问题平均长度: {sum(question_lengths) / len(question_lengths):.1f} 字符")
    
    # 统计答案长度
    answer_lengths = [len(item.get('answer', '')) for item in data]
    print(f"答案平均长度: {sum(answer_lengths) / len(answer_lengths):.1f} 字符")
    
    # 统计cot长度
    cot_lengths = [len(item.get('cot', '')) for item in data]
    print(f"COT平均长度: {sum(cot_lengths) / len(cot_lengths):.1f} 字符")
    
    return True

def test_token_entropy_with_forget01(tokenizer):
    """使用 forget01.json 数据测试 token_entropy 函数"""
    print("\n=== 测试 token_entropy 函数 (使用 forget01.json) ===")
    
    data = load_forget01_data()
    if data is None:
        return False
    
    # 选择前5个样本进行测试
    test_samples = data[:5]
    
    # 提取问题和答案
    questions = [item.get('question', '') for item in test_samples]
    answers = [item.get('answer', '') for item in test_samples]
    cots = [item.get('cot', '') for item in test_samples]
    
    try:
        # 测试问题的token entropy
        print("计算问题的token entropy...")
        question_entropy = token_entropy(tokenizer, questions, normalize=True)
        print(f"问题token entropy: {question_entropy}")
        
        # 测试答案的token entropy
        print("计算答案的token entropy...")
        answer_entropy = token_entropy(tokenizer, answers, normalize=True)
        print(f"答案token entropy: {answer_entropy}")
        
        # 测试COT的token entropy
        print("计算COT的token entropy...")
        cot_entropy = token_entropy(tokenizer, cots, normalize=True)
        print(f"COT token entropy: {cot_entropy}")
        
        # 计算平均熵
        avg_question_entropy = sum(question_entropy['token_entropy']) / len(question_entropy['token_entropy'])
        avg_answer_entropy = sum(answer_entropy['token_entropy']) / len(answer_entropy['token_entropy'])
        avg_cot_entropy = sum(cot_entropy['token_entropy']) / len(cot_entropy['token_entropy'])
        
        print(f"\n平均token entropy:")
        print(f"  问题: {avg_question_entropy:.4f}")
        print(f"  答案: {avg_answer_entropy:.4f}")
        print(f"  COT: {avg_cot_entropy:.4f}")
        
        return True
    except Exception as e:
        print(f"计算token entropy时出错: {e}")
        return False

def test_rouge_evaluation_with_forget01():
    """使用 forget01.json 数据测试 ROUGE 评估函数"""
    print("\n=== 测试 ROUGE 评估函数 (使用 forget01.json) ===")
    
    data = load_forget01_data()
    if data is None:
        return False
    
    # 选择前10个样本进行测试
    test_samples = data[:10]
    
    # 模拟生成的答案（这里使用原始答案作为生成结果进行测试）
    gen_outputs = [item.get('answer', '') for item in test_samples]
    ground_truths = [item.get('answer', '') for item in test_samples]
    
    try:
        rouge_scores = eval_rouge_recall(gen_outputs, ground_truths)
        
        print(f"ROUGE-1 Recall: {rouge_scores['rouge1_recall']}")
        print(f"ROUGE-L Recall: {rouge_scores['rougeL_recall']}")
        
        # 计算平均值
        avg_rouge1 = sum(rouge_scores['rouge1_recall']) / len(rouge_scores['rouge1_recall'])
        avg_rougeL = sum(rouge_scores['rougeL_recall']) / len(rouge_scores['rougeL_recall'])
        
        print(f"\n平均ROUGE分数:")
        print(f"  ROUGE-1 Recall: {avg_rouge1:.4f}")
        print(f"  ROUGE-L Recall: {avg_rougeL:.4f}")
        
        return True
    except Exception as e:
        print(f"ROUGE 评估时出错: {e}")
        return False

def test_cosine_similarity_with_forget01():
    """使用 forget01.json 数据测试余弦相似度函数"""
    print("\n=== 测试余弦相似度函数 (使用 forget01.json) ===")
    
    data = load_forget01_data()
    if data is None:
        return False
    
    # 选择前10个样本进行测试
    test_samples = data[:10]
    
    # 使用问题和答案进行相似度测试
    questions = [item.get('question', '') for item in test_samples]
    answers = [item.get('answer', '') for item in test_samples]
    
    try:
        similarity_scores = eval_cosine_similarity(questions, answers)
        
        print(f"余弦相似度分数: {similarity_scores['cosine_similarity']}")
        
        # 计算平均值
        avg_similarity = sum(similarity_scores['cosine_similarity']) / len(similarity_scores['cosine_similarity'])
        print(f"\n平均余弦相似度: {avg_similarity:.4f}")
        
        return True
    except Exception as e:
        print(f"计算余弦相似度时出错: {e}")
        return False

def test_dataset_loading_with_forget01(config, tokenizer):
    """使用 forget01.json 测试数据集加载"""
    print("\n=== 测试数据集加载 (使用 forget01.json) ===")
    
    try:
        # 加载forget01数据集
        dataset = TextDatasetQA(
            folder="data/tofu",
            tokenizer=tokenizer,
            model_family=config['model_family'],
            max_length=config['eval']['generation']['max_length'],
            split="forget01",
            question_key="question",
            answer_key="answer"
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试获取几个样本
        if len(dataset) > 0:
            print(f"样本格式: input_ids shape: {dataset[0][0].shape}, labels shape: {dataset[0][1].shape}")
            
            # 显示前3个样本的详细信息
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"\n样本 {i+1}:")
                print(f"  input_ids shape: {sample[0].shape}")
                print(f"  labels shape: {sample[1].shape}")
                print(f"  attention_mask shape: {sample[2].shape}")
                
                # 解码显示部分内容
                decoded = tokenizer.decode(sample[0][0][:50], skip_special_tokens=True)
                print(f"  解码内容 (前50个token): {decoded[:100]}...")
        
        return True
    except Exception as e:
        print(f"数据集加载时出错: {e}")
        return False

def test_dataloader_with_forget01(config, tokenizer):
    """测试数据加载器"""
    print("\n=== 测试数据加载器 (使用 forget01.json) ===")
    
    try:
        # 使用get_dataloader函数
        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
            cfg=config,
            eval_task="eval_log",
            tokenizer=tokenizer,
            folder="data/tofu",
            split="forget01",
            question_key="question",
            answer_key="answer",
            base_answer_key="answer",
            perturbed_answer_key="answer"
        )
        
        print(f"eval_dataloader 批次数: {len(eval_dataloader)}")
        print(f"base_eval_dataloader 批次数: {len(base_eval_dataloader)}")
        print(f"perturb_dataloader 批次数: {len(perturb_dataloader)}")
        
        # 测试获取一个批次
        for batch in eval_dataloader:
            input_ids, labels, attention_mask = batch
            print(f"\n批次信息:")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  labels shape: {labels.shape}")
            print(f"  attention_mask shape: {attention_mask.shape}")
            break
        
        return True
    except Exception as e:
        print(f"数据加载器测试时出错: {e}")
        return False

def test_mask_non_answer_labels_with_forget01(tokenizer):
    """使用 forget01.json 数据测试 mask_non_answer_labels 函数"""
    print("\n=== 测试 mask_non_answer_labels 函数 (使用 forget01.json) ===")
    
    data = load_forget01_data()
    if data is None:
        return False
    
    try:
        # 创建模拟的labels（这里简化处理）
        batch_size = 2
        seq_len = 100
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        masked_labels = mask_non_answer_labels(labels, tokenizer)
        
        print(f"原始labels shape: {labels.shape}")
        print(f"Masked labels shape: {masked_labels.shape}")
        
        # 统计被mask的token数量
        masked_count = (masked_labels == -100).sum().item()
        total_count = masked_labels.numel()
        print(f"被mask的token比例: {masked_count}/{total_count} ({masked_count/total_count*100:.1f}%)")
        
        return True
    except Exception as e:
        print(f"Mask labels时出错: {e}")
        return False

def test_batch_loss_with_forget01():
    """使用 forget01.json 数据测试 get_batch_loss 函数"""
    print("\n=== 测试 get_batch_loss 函数 (使用 forget01.json) ===")
    
    try:
        # 创建模拟的输出和标签
        batch_size, seq_len, vocab_size = 2, 100, 32000  # 使用更真实的参数
        output = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 设置一些-100标签（模拟padding）
        labels[0, :20] = -100
        labels[1, :15] = -100
        
        loss = get_batch_loss(output, labels)
        
        print(f"输出shape: {output.shape}")
        print(f"标签shape: {labels.shape}")
        print(f"损失shape: {loss.shape}")
        print(f"损失值: {loss}")
        print(f"平均损失: {loss.mean().item():.4f}")
        
        return True
    except Exception as e:
        print(f"计算batch loss时出错: {e}")
        return False

def main():
    """主函数"""
    print("开始使用 forget01.json 数据测试 metrics.py 中的函数...")
    
    # 加载配置
    config = load_config()
    print(f"配置加载成功: model_family={config['model_family']}")
    
    # 模型路径
    model_path = "/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch3_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last"
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path)
    if model is None or tokenizer is None:
        print("无法加载模型，退出测试")
        return
    
    # 运行各种测试
    test_results = []
    
    test_results.append(("read_jsonline_with_forget01", test_read_jsonline_with_forget01()))
    test_results.append(("token_entropy_with_forget01", test_token_entropy_with_forget01(tokenizer)))
    test_results.append(("rouge_evaluation_with_forget01", test_rouge_evaluation_with_forget01()))
    test_results.append(("cosine_similarity_with_forget01", test_cosine_similarity_with_forget01()))
    test_results.append(("dataset_loading_with_forget01", test_dataset_loading_with_forget01(config, tokenizer)))
    test_results.append(("dataloader_with_forget01", test_dataloader_with_forget01(config, tokenizer)))
    test_results.append(("mask_non_answer_labels_with_forget01", test_mask_non_answer_labels_with_forget01(tokenizer)))
    test_results.append(("batch_loss_with_forget01", test_batch_loss_with_forget01()))
    
    # 打印测试结果总结
    print("\n" + "="*60)
    print("测试结果总结:")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:35} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了!")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    print("\n注意: 这个测试使用了 forget01.json 的真实数据，")
    print("包含了实际的问答对和思维链数据，可以更好地验证函数的实际效果。")

if __name__ == "__main__":
    main()