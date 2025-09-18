import os
import shutil
import torch
import hydra
import transformers
from utils import get_model_identifiers_from_yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
import datasets
from datasets import load_dataset, Dataset
import json
from omegaconf import DictConfig
import torch.distributed as dist
import sys
from transformers import GenerationConfig
import re
from rouge_score import rouge_scorer
from scipy.stats import hmean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline
import signal
import time
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter
import logging
import pandas as pd
import datetime

# 检查是否安装了openpyxl
try:
    import openpyxl
except ImportError:
    print("警告: 未安装openpyxl库，Excel功能可能无法正常工作。请运行: pip install openpyxl")
# 确保能找到本地的trl库
current_dir = os.path.dirname(os.path.abspath(__file__))
trl_path = os.path.join(current_dir, 'trl')
if trl_path not in sys.path:
    sys.path.insert(0, trl_path)

from trl import GRPOConfig, GRPOTrainer
from utils import get_model_identifiers_from_yaml, find_all_linear_names


class CustomGRPOTrainer(GRPOTrainer):
    """自定义GRPO训练器，增加TensorBoard记录功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tb_writer = None
        
    def setup_tensorboard(self, log_dir):
        """设置TensorBoard写入器"""
        self.tb_writer = SummaryWriter(log_dir)
        
    def log(self, logs, start_time=None):
        """重写日志方法，添加额外的TensorBoard记录"""
        super().log(logs, start_time)
        
        if self.tb_writer is not None and self.state.global_step > 0:
            # 记录基础训练指标
            if "train/learning_rate" in logs:
                self.tb_writer.add_scalar("learning_rate", logs["train/learning_rate"], self.state.global_step)
            
            if "train/loss" in logs:
                self.tb_writer.add_scalar("loss/total_loss", logs["train/loss"], self.state.global_step)
            
            # 记录GRPO特有的指标
            if "train/reward" in logs:
                self.tb_writer.add_scalar("grpo/reward", logs["train/reward"], self.state.global_step)
            
            if "train/kl_divergence" in logs:
                self.tb_writer.add_scalar("grpo/kl_divergence", logs["train/kl_divergence"], self.state.global_step)
                
            # 记录各个奖励分量（如果可用）
            for i, reward_name in enumerate(["rouge_recall_forget", "cosine_similarity_forget", "quality"]):
                if f"train/reward_{i}" in logs:
                    self.tb_writer.add_scalar(f"rewards/{reward_name}", logs[f"train/reward_{i}"], self.state.global_step)
            
            # 记录梯度范数
            if "train/grad_norm" in logs:
                self.tb_writer.add_scalar("optimization/grad_norm", logs["train/grad_norm"], self.state.global_step)
            
            # 刷新TensorBoard
            self.tb_writer.flush()
    
    def on_train_end(self):
        """训练结束时关闭TensorBoard写入器"""
        super().on_train_end()
        if self.tb_writer is not None:
            self.tb_writer.close()



def load_forget_dataset(data_path, split, task_id):
    """加载遗忘数据集"""
    file_path = f"{data_path}/{split}.json"
    
    # 使用datasets.load_dataset处理JSONL格式文件
    data = datasets.load_dataset('json', data_files=file_path, split='train')
    
    # 根据task_id过滤数据
    if task_id != -1:
        data = data.filter(lambda x: int(x['task_id']) == task_id)
    
    # 转换为列表格式以保持兼容性
    return list(data)


def create_grpo_dataset(forget_data, tokenizer, model_configs):
    """创建GRPO训练所需的数据集格式"""
    grpo_data = []
    
    # 处理遗忘数据 - 转换为prompt格式
    for item in forget_data:
        question = item['question']
        
        # 构建prompt
        begin_of_sentence_token = model_configs['begin_of_sentence_tag']
        end_of_sentence_token = model_configs['end_of_sentence_tag']
        question_start_token = model_configs['question_start_tag']
        question_end_token = model_configs['question_end_tag']
        think_start_token = model_configs['think_start_tag']
        
        prompt = f"{begin_of_sentence_token}{question}{think_start_token}"
        
        grpo_data.append({
            'prompt': prompt,
            'ground_truth_answer': item['answer'],
            'ground_truth_cot': item['cot'],
            'data_type': 'forget',
            'task_id': item['task_id']
        })

    
    # 打印前五条prompt
    print("\n前五条prompt:")
    print("=" * 80)
    for i, item in enumerate(grpo_data[:5]):
        print(f"第{i+1}条 prompt:")
        print(f"任务ID: {item['task_id']}")
        print(f"Prompt: {item['prompt']}")
        print(f"真实答案: {item['ground_truth_answer']}")
        print(f"CoT推理: {item['ground_truth_cot']}")
        print("-" * 60)
    return Dataset.from_list(grpo_data)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# 使用本地下载的SentenceTransformer模型
local_sentence_model_path = "/ljq/rtofu/local_models/paraphrase-MiniLM-L6-v2"
modelst = SentenceTransformer(local_sentence_model_path, device=torch.device('cuda'))



def create_rouge_recall_forget_reward_function(output_dir):

    def rouge_recall_forget_reward(completions, ground_truth_answer, ground_truth_cot, data_type, **kwargs):
        rewards = []
        completions = [
            re.sub(r'<｜end▁of▁sentence｜>+', '', s.split("</think>\n\n", 1)[1]).strip()
            if "</think>\n\n" in s else s
            for s in completions
        ]
        # 创建Excel记录数据
        excel_data = []
        
        for gen, gt in zip(completions, ground_truth_answer):
            rouge_scores = scorer.score(gt, gen)
            reward_value = 1-rouge_scores['rougeL'].recall
            rewards.append(reward_value)
            
            # 记录到Excel数据中
            excel_data.append({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reward_type': 'rouge_recall_forget',
                'completion': gen,
                'ground_truth_answer': gt,
                'reward_value': reward_value,
                'rouge_recall': rouge_scores['rougeL'].recall
            })
        
        # 保存到Excel文件 - 只在rank 0进程中保存
        if excel_data and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            reward_log_dir = os.path.join(output_dir, 'reward_log')
            os.makedirs(reward_log_dir, exist_ok=True)
            excel_file = os.path.join(reward_log_dir, 'reward_logs_rouge.xlsx')
            try:
                # 尝试读取现有文件
                if os.path.exists(excel_file):
                    existing_df = pd.read_excel(excel_file, engine='openpyxl')
                    new_df = pd.DataFrame(excel_data)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = pd.DataFrame(excel_data)
                
                # 保存到Excel文件
                combined_df.to_excel(excel_file, index=False, engine='openpyxl')
            except Exception as e:
                print(f"保存Rouge奖励记录到Excel时出错: {e}")
        
        return rewards
    return rouge_recall_forget_reward


def create_cosine_similarity_forget_reward_function(output_dir):

    def cosine_similarity_forget_reward(completions, ground_truth_answer, ground_truth_cot, data_type, **kwargs):
        rewards = []
        completions = [
            re.sub(r'<｜end▁of▁sentence｜>+', '', s.split("</think>\n\n", 1)[1]).strip()
            if "</think>\n\n" in s else s
            for s in completions
        ]
        
        # 创建Excel记录数据
        excel_data = []
        
        with torch.no_grad():
            for gen, gt in zip(completions, ground_truth_answer):
                gen_embedding = modelst.encode(gen, show_progress_bar=False)
                gt_embedding = modelst.encode(gt, show_progress_bar=False)
                cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
                reward_value = 1-float(max(0, cosine_sim))
                rewards.append(reward_value)
                
                # 记录到Excel数据中
                excel_data.append({
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'reward_type': 'cosine_similarity_forget',
                    'completion': gen,
                    'ground_truth_answer': gt,
                    'reward_value': reward_value,
                    'cosine_similarity': float(cosine_sim)
                })
        
        # 保存到Excel文件 - 只在rank 0进程中保存
        if excel_data and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            reward_log_dir = os.path.join(output_dir, 'reward_log')
            os.makedirs(reward_log_dir, exist_ok=True)
            excel_file = os.path.join(reward_log_dir, 'reward_logs_cosine.xlsx')
            try:
                # 尝试读取现有文件
                if os.path.exists(excel_file):
                    existing_df = pd.read_excel(excel_file, engine='openpyxl')
                    new_df = pd.DataFrame(excel_data)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = pd.DataFrame(excel_data)
                
                # 保存到Excel文件
                combined_df.to_excel(excel_file, index=False, engine='openpyxl')
            except Exception as e:
                print(f"保存余弦相似度奖励记录到Excel时出错: {e}")
        
        return rewards
    return cosine_similarity_forget_reward



def create_quality_reward_function(output_dir):
    """创建用于评估回答质量的奖励函数"""
    
    def quality_reward(completions, ground_truth_answer=None, **kwargs):
        """
        质量奖励函数：评估回答的一般质量
        """
        rewards = []
        pattern = re.compile(r'</think>[\s\S]*?')
        
        # 创建Excel记录数据
        excel_data = []

        for i, completion in enumerate(completions):
            if bool(pattern.search(completion)):
                reward_value = 1
            else:
                reward_value = 0
            rewards.append(reward_value)
            
            # 记录到Excel数据中
            gt_answer = ground_truth_answer[i] if ground_truth_answer and i < len(ground_truth_answer) else "N/A"
            excel_data.append({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reward_type': 'quality',
                'completion': completion,
                'ground_truth_answer': gt_answer,
                'reward_value': reward_value,
                'has_think_tag': bool(pattern.search(completion))
            })
        
        # 保存到Excel文件 - 只在rank 0进程中保存
        if excel_data and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            reward_log_dir = os.path.join(output_dir, 'reward_log')
            os.makedirs(reward_log_dir, exist_ok=True)
            excel_file = os.path.join(reward_log_dir, 'reward_logs_quality.xlsx')
            try:
                # 尝试读取现有文件
                if os.path.exists(excel_file):
                    existing_df = pd.read_excel(excel_file, engine='openpyxl')
                    new_df = pd.DataFrame(excel_data)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = pd.DataFrame(excel_data)
                
                # 保存到Excel文件
                combined_df.to_excel(excel_file, index=False, engine='openpyxl')
            except Exception as e:
                print(f"保存质量奖励记录到Excel时出错: {e}")
        
        return rewards
    return quality_reward


@hydra.main(version_base=None, config_path="config", config_name="tofu")
def main(cfg: DictConfig):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 基本设置
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"开始GRPO训练，任务ID: {cfg.task_id}")
    
    # 数据加载
    forget_data = load_forget_dataset(cfg.data_path, cfg.split, cfg.task_id)

    print(f"遗忘数据数量: {len(forget_data)}")
    
    # 模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # GRPO需要左对齐
    
    # 获取模型配置
    
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    model_configs = get_model_identifiers_from_yaml(cfg.model_family)
    
    # 创建GRPO数据集
    train_dataset = create_grpo_dataset(forget_data, tokenizer, model_configs)
    
    # 设置训练参数
    output_dir_1 = cfg.save_dir.replace('seed_', 'grpo_seed_')
    output_dir = os.path.join(output_dir_1, f"unlearn_times_{cfg.task_id}")
    
    # 设置TensorBoard日志目录
    tensorboard_log_dir = os.path.join(output_dir, "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # GRPO配置
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        max_steps=-1,
        warmup_steps=0,
        save_steps=13,
        save_only_model=True,
        eval_strategy="no",
        bf16=True,
        deepspeed="config/ds_config/llama3.json" if not cfg.use_LoRA else None,
        remove_unused_columns=False,
        # TensorBoard日志配置
        logging_dir=tensorboard_log_dir,
        logging_steps=1,  # 每1步记录一次
        report_to="tensorboard",
        # GRPO特有参数 - 改进的生成配置
        max_prompt_length=cfg.grpo.max_prompt_length,
        max_completion_length=cfg.grpo.max_completion_length,
        num_generations=cfg.grpo.num_generations,  # 每个prompt生成4个候选
        temperature=cfg.grpo.temperature,
        top_p=cfg.grpo.top_p,
        # 额外的生成控制参数
     
        repetition_penalty=1.1,  # 重复惩罚
        
        # 奖励权重
        reward_weights=[1,2,0.2],  # 遗忘奖励权重更高，质量奖励权重较低
    )
    
    # 打印GRPO配置信息
    print(f"GRPO配置:")
    print(f"  最大prompt长度: {cfg.grpo.max_prompt_length}")
    print(f"  最大completion长度: {cfg.grpo.max_completion_length}")
    print(f"  最大新token数: {cfg.grpo.max_completion_length}")
    print(f"  生成数量: {cfg.grpo.num_generations}")
    print(f"  温度: {cfg.grpo.temperature}")
    print(f"  Top-p: {cfg.grpo.top_p}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  TensorBoard日志目录: {tensorboard_log_dir}")
    last_checkpoint_dir = os.path.join(output_dir_1, f"unlearn_times_{cfg.task_id-1}",f"checkpoint-65")
    model_path = cfg.model_path if cfg.task_id == 1 else last_checkpoint_dir
    
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling.setdefault("type", "linear")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        config=config,
        attn_implementation='flash_attention_2',
        trust_remote_code=True,
    )
    
    # 配置LoRA
    if cfg.use_LoRA:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=find_all_linear_names(model),
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            lora_dropout=cfg.LoRA.dropout,
        )
        model = get_peft_model(model, peft_config)
    
    # 创建奖励函数
    rouge_recall_forget_reward_func = create_rouge_recall_forget_reward_function(output_dir)
    cosine_similarity_forget_reward_func = create_cosine_similarity_forget_reward_function(output_dir)
    quality_reward_func = create_quality_reward_function(output_dir)
    
    model.generation_config = GenerationConfig(
    max_new_tokens=cfg.grpo.max_completion_length,                   # 最长生成
    eos_token_id=tokenizer.eos_token_id,  # 停止符
    pad_token_id=tokenizer.eos_token_id   # 避免报错
    )
    # 初始化自定义GRPO训练器
    trainer = CustomGRPOTrainer(
        model=model,
        reward_funcs=[rouge_recall_forget_reward_func, cosine_similarity_forget_reward_func, quality_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config if cfg.use_LoRA else None,
    )
    
    # 设置TensorBoard
    trainer.setup_tensorboard(tensorboard_log_dir)
    
    print('开始GRPO训练...')
    trainer.train()
    
    if torch.distributed.get_rank() == 0:
        #trainer.model.save_pretrained(output_dir, safe_serialization=False)
        #trainer.tokenizer.save_pretrained(output_dir)
        print(f"训练完成！模型已保存到: {output_dir}")
        print(f"TensorBoard日志已保存到: {tensorboard_log_dir}")
        print(f"要查看训练过程，请运行: tensorboard --logdir {tensorboard_log_dir}")
        print(f"然后在浏览器中打开 http://localhost:6006")


if __name__ == "__main__":
    main()
