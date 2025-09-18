#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励值计算脚本
专门用于计算遗忘数据集样本的奖励值
"""

import os
import torch
import json
import re
import argparse
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class RewardCalculator:
    """奖励值计算器"""
    
    def __init__(self, model_path: str, sentence_model_path: str = None):
        """
        初始化奖励计算器
        
        Args:
            model_path: 语言模型路径
            sentence_model_path: 句子嵌入模型路径，默认使用本地模型
        """
        print("正在初始化奖励计算器...")
        
        # 加载语言模型和分词器
        print(f"正在加载语言模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            trust_remote_code=True,
            device_map="auto"
        )
        
        # 设置生成配置
        self.model.generation_config = GenerationConfig(
            max_new_tokens=512,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # 加载句子嵌入模型用于余弦相似度计算
        if sentence_model_path is None:
            sentence_model_path = "/ljq/rtofu/local_models/paraphrase-MiniLM-L6-v2"
        
        print(f"正在加载句子嵌入模型: {sentence_model_path}")
        self.sentence_model = SentenceTransformer(
            sentence_model_path, 
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 初始化ROUGE评分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        print("奖励计算器初始化完成!")
    
    def load_forget_dataset(self, data_path: str, split: str = "train", task_id: int = -1) -> List[Dict]:
        """
        加载遗忘数据集
        
        Args:
            data_path: 数据集路径
            split: 数据集分割（train/test/val）
            task_id: 任务ID，-1表示加载所有任务
            
        Returns:
            数据集列表
        """
        print(f"正在加载遗忘数据集: {data_path}/{split}.json")
        
        file_path = f"{data_path}/{split}.json"
        data = load_dataset('json', data_files=file_path, split='train')
        
        # 根据task_id过滤数据
        if task_id != -1:
            data = data.filter(lambda x: int(x['task_id']) == task_id)
            print(f"已过滤task_id={task_id}的数据")
        
        dataset_list = list(data)
        print(f"加载了 {len(dataset_list)} 个样本")
        return dataset_list
    
    def create_prompt(self, question: str, model_configs: Dict[str, str]) -> str:
        """
        创建输入prompt
        
        Args:
            question: 问题文本
            model_configs: 模型配置字典
            
        Returns:
            格式化的prompt
        """
        begin_of_sentence_token = model_configs.get('begin_of_sentence_tag', '<|begin_of_sentence|>')
        question_start_token = model_configs.get('question_start_tag', '<|question|>')
        question_end_token = model_configs.get('question_end_tag', '<|/question|>')
        think_start_token = model_configs.get('think_start_tag', '<think>')
        
        prompt = f"{begin_of_sentence_token}{question_start_token}{question}{question_end_token}{think_start_token}"
        return prompt
    
    def generate_responses(self, prompts: List[str], num_generations: int = 3) -> List[List[str]]:
        """
        为给定prompts生成多个回复
        
        Args:
            prompts: prompt列表
            num_generations: 每个prompt生成的回复数量
            
        Returns:
            每个prompt对应的多个回复列表
        """
        all_responses = []
        
        print(f"正在为 {len(prompts)} 个prompt生成回复，每个生成 {num_generations} 个...")
        
        for i, prompt in enumerate(tqdm(prompts, desc="生成回复")):
            responses = []
            
            for gen_idx in range(num_generations):
                # 编码输入
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                
                # 移动到GPU
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # 设置随机种子以获得多样性
                torch.manual_seed(42 + gen_idx + i * num_generations)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    # 解码生成的文本
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    responses.append(generated_text)
            
            all_responses.append(responses)
        
        return all_responses
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        从生成的回复中提取答案部分
        
        Args:
            response: 完整的生成回复
            
        Returns:
            提取的答案
        """
        # 移除结束符号
        response = re.sub(r'<｜end▁of▁sentence｜>+', '', response)
        
        # 如果包含</think>，提取think标签后的内容
        if "</think>\n\n" in response:
            answer = response.split("</think>\n\n", 1)[1].strip()
        elif "</think>" in response:
            answer = response.split("</think>", 1)[1].strip()
        else:
            answer = response.strip()
        
        return answer
    
    def calculate_rouge_recall_reward(self, completions: List[str], ground_truth_answers: List[str]) -> List[float]:
        """
        计算ROUGE召回奖励（遗忘奖励）
        
        Args:
            completions: 生成的完整回复列表
            ground_truth_answers: 标准答案列表
            
        Returns:
            奖励值列表
        """
        rewards = []
        
        # 提取答案
        extracted_answers = [self.extract_answer_from_response(comp) for comp in completions]
        
        for gen_answer, gt_answer in zip(extracted_answers, ground_truth_answers):
            rouge_scores = self.rouge_scorer.score(gt_answer, gen_answer)
            # 遗忘奖励：1 - ROUGE召回率（越不像标准答案，奖励越高）
            reward = 1 - rouge_scores['rougeL'].recall
            rewards.append(reward)
        
        return rewards
    
    def calculate_cosine_similarity_reward(self, completions: List[str], ground_truth_answers: List[str]) -> List[float]:
        """
        计算余弦相似度奖励（遗忘奖励）
        
        Args:
            completions: 生成的完整回复列表
            ground_truth_answers: 标准答案列表
            
        Returns:
            奖励值列表
        """
        rewards = []
        
        # 提取答案
        extracted_answers = [self.extract_answer_from_response(comp) for comp in completions]
        
        with torch.no_grad():
            for gen_answer, gt_answer in zip(extracted_answers, ground_truth_answers):
                # 计算嵌入
                gen_embedding = self.sentence_model.encode(gen_answer, show_progress_bar=False)
                gt_embedding = self.sentence_model.encode(gt_answer, show_progress_bar=False)
                
                # 计算余弦相似度
                cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
                
                # 遗忘奖励：1 - 余弦相似度（越不相似，奖励越高）
                reward = 1 - float(max(0, cosine_sim))
                rewards.append(reward)
        
        return rewards
    
    def calculate_quality_reward(self, completions: List[str]) -> List[float]:
        """
        计算质量奖励
        
        Args:
            completions: 生成的完整回复列表
            
        Returns:
            奖励值列表
        """
        rewards = []
        pattern = re.compile(r'</think>[\s\S]*?')
        
        for completion in completions:
            if bool(pattern.search(completion)):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    def calculate_all_rewards(self, responses_batch: List[List[str]], 
                            ground_truth_answers: List[str],
                            ground_truth_cots: List[str]) -> Dict[str, List[List[float]]]:
        """
        计算所有类型的奖励值
        
        Args:
            responses_batch: 每个样本的多个回复 [[resp1_1, resp1_2, ...], [resp2_1, resp2_2, ...], ...]
            ground_truth_answers: 标准答案列表
            ground_truth_cots: 标准推理过程列表
            
        Returns:
            包含所有奖励类型的字典
        """
        all_rewards = {
            'rouge_recall_rewards': [],
            'cosine_similarity_rewards': [],
            'quality_rewards': []
        }
        
        print("正在计算奖励值...")
        
        for i, (responses, gt_answer, gt_cot) in enumerate(zip(responses_batch, ground_truth_answers, ground_truth_cots)):
            # 为当前样本的所有回复计算奖励
            rouge_rewards = self.calculate_rouge_recall_reward(responses, [gt_answer] * len(responses))
            cosine_rewards = self.calculate_cosine_similarity_reward(responses, [gt_answer] * len(responses))
            quality_rewards = self.calculate_quality_reward(responses)
            
            all_rewards['rouge_recall_rewards'].append(rouge_rewards)
            all_rewards['cosine_similarity_rewards'].append(cosine_rewards)
            all_rewards['quality_rewards'].append(quality_rewards)
        
        return all_rewards
    
    def calculate_average_rewards(self, all_rewards: Dict[str, List[List[float]]]) -> Dict[str, float]:
        """
        计算平均奖励值
        
        Args:
            all_rewards: 所有奖励值字典
            
        Returns:
            平均奖励值字典
        """
        avg_rewards = {}
        
        for reward_type, rewards_list in all_rewards.items():
            # 展平所有奖励值并计算平均值
            all_values = []
            for sample_rewards in rewards_list:
                all_values.extend(sample_rewards)
            
            avg_rewards[reward_type] = np.mean(all_values) if all_values else 0.0
            avg_rewards[f'{reward_type}_per_sample'] = [np.mean(sample_rewards) for sample_rewards in rewards_list]
        
        return avg_rewards

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算遗忘数据集样本的奖励值")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="train", help="数据集分割")
    parser.add_argument("--task_id", type=int, default=-1, help="任务ID，-1表示所有任务")
    parser.add_argument("--num_generations", type=int, default=3, help="每个样本生成的回复数量")
    parser.add_argument("--max_samples", type=int, default=-1, help="最大样本数量，-1表示所有样本")
    parser.add_argument("--output_file", type=str, default="reward_results.json", help="输出结果文件")
    parser.add_argument("--sentence_model_path", type=str, default=None, help="句子嵌入模型路径")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("奖励值计算脚本")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"数据分割: {args.split}")
    print(f"任务ID: {args.task_id}")
    print(f"每样本生成数: {args.num_generations}")
    print(f"最大样本数: {args.max_samples}")
    print("-" * 80)
    
    # 初始化奖励计算器
    calculator = RewardCalculator(args.model_path, args.sentence_model_path)
    
    # 加载数据集
    dataset = calculator.load_forget_dataset(args.data_path, args.split, args.task_id)
    
    # 限制样本数量
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
        print(f"限制样本数量为: {len(dataset)}")
    
    # 模型配置
    model_configs = {
        'begin_of_sentence_tag': '<|begin_of_sentence|>',
        'question_start_tag': '<|question|>',
        'question_end_tag': '<|/question|>',
        'think_start_tag': '<think>'
    }
    
    # 创建prompts
    print("创建prompts...")
    prompts = []
    ground_truth_answers = []
    ground_truth_cots = []
    
    for item in dataset:
        prompt = calculator.create_prompt(item['question'], model_configs)
        prompts.append(prompt)
        ground_truth_answers.append(item['answer'])
        ground_truth_cots.append(item.get('cot', ''))
    
    # 生成回复
    responses_batch = calculator.generate_responses(prompts, args.num_generations)
    
    # 计算奖励值
    all_rewards = calculator.calculate_all_rewards(responses_batch, ground_truth_answers, ground_truth_cots)
    
    # 计算平均奖励值
    avg_rewards = calculator.calculate_average_rewards(all_rewards)
    
    # 准备结果
    results = {
        'config': {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'split': args.split,
            'task_id': args.task_id,
            'num_generations': args.num_generations,
            'num_samples': len(dataset),
        },
        'average_rewards': {k: v for k, v in avg_rewards.items() if not k.endswith('_per_sample')},
        'per_sample_rewards': {k: v for k, v in avg_rewards.items() if k.endswith('_per_sample')},
        'detailed_rewards': all_rewards,
        'sample_details': []
    }
    
    # 添加详细的样本信息
    for i, (item, responses, rouge_rewards, cosine_rewards, quality_rewards) in enumerate(zip(
        dataset, responses_batch, 
        all_rewards['rouge_recall_rewards'],
        all_rewards['cosine_similarity_rewards'], 
        all_rewards['quality_rewards']
    )):
        sample_detail = {
            'sample_id': i,
            'question': item['question'],
            'ground_truth_answer': item['answer'],
            'generated_responses': responses,
            'extracted_answers': [calculator.extract_answer_from_response(resp) for resp in responses],
            'rouge_recall_rewards': rouge_rewards,
            'cosine_similarity_rewards': cosine_rewards,
            'quality_rewards': quality_rewards,
            'avg_rouge_reward': np.mean(rouge_rewards),
            'avg_cosine_reward': np.mean(cosine_rewards),
            'avg_quality_reward': np.mean(quality_rewards)
        }
        results['sample_details'].append(sample_detail)
    
    # 保存结果
    print(f"保存结果到: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印结果摘要
    print("\n" + "=" * 80)
    print("奖励计算结果摘要")
    print("=" * 80)
    print(f"总样本数: {len(dataset)}")
    print(f"每样本生成数: {args.num_generations}")
    print(f"总生成数: {len(dataset) * args.num_generations}")
    print()
    print("平均奖励值:")
    for reward_type, value in results['average_rewards'].items():
        print(f"  {reward_type}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("计算完成!")
    print("=" * 80)

if __name__ == "__main__":
    main() 