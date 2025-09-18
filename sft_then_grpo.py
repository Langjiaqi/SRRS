import os
import shutil
import warnings
from pathlib import Path
import sys

import datasets
import hydra
import torch
import transformers
from omegaconf import OmegaConf, DictConfig
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
from datasets import Dataset
import re
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 本地模块导入
from dataset import TextForgetDatasetQA, dataset_to_json, custom_data_collator_forget
from trainer import CustomTrainerForgetting
from utils import get_model_identifiers_from_yaml, set_random_seed, find_all_linear_names

# 确保能找到本地的trl库
current_dir = os.path.dirname(os.path.abspath(__file__))
trl_path = os.path.join(current_dir, 'trl')
if trl_path not in sys.path:
    sys.path.insert(0, trl_path)

from trl import GRPOConfig, GRPOTrainer

warnings.filterwarnings('ignore')

# 全局变量用于GRPO
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
local_sentence_model_path = "/ljq/rtofu/local_models/paraphrase-MiniLM-L6-v2"
modelst = SentenceTransformer(local_sentence_model_path, device=torch.device('cuda'))


def get_task_data(data_path, split, task_id, unlearned_tasks, curr_save_dir):
    """获取任务数据（SFT阶段使用）"""
    local_rank = int(os.environ['LOCAL_RANK'])
    forget_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split + '.json'), split='train')
    forget_pertrubed_data = datasets.load_dataset('json', data_files=os.path.join(data_path, split + '_perturbed.json'),
                                                  split='train')

    retain_split = "retain" + str(100 - min(10 * int(split.replace("forget", "")), 90)).zfill(2)
    retain_data = datasets.load_dataset('json', data_files=os.path.join(data_path, retain_split + '.json'),
                                        split='train')

    forget_retain_data = forget_data.filter(lambda x: int(x['task_id']) not in unlearned_tasks)
    curr_forget_data = forget_data.filter(lambda x: int(x['task_id']) == task_id)

    curr_retain_data = datasets.concatenate_datasets([retain_data, forget_retain_data])

    curr_forget_perturbed_data = forget_pertrubed_data.filter(lambda x: int(x['task_id']) == task_id)

    if local_rank == 0:
        curr_data_path = os.path.join(curr_save_dir, 'task_data')
        os.makedirs(curr_data_path, exist_ok=True)
        dataset_to_json(curr_forget_data, os.path.join(
            curr_data_path, 'forget.json'))
        dataset_to_json(curr_forget_perturbed_data, os.path.join(
            curr_data_path, 'forget_perturbed.json'))
        dataset_to_json(curr_retain_data, os.path.join(
            curr_data_path, 'retain.json'))

    return curr_forget_data, curr_retain_data


def load_forget_dataset(data_path, split, task_id):
    """加载遗忘数据集（GRPO阶段使用）"""
    file_path = f"{data_path}/{split}.json"
    
    data = datasets.load_dataset('json', data_files=file_path, split='train')
    
    if task_id != -1:
        data = data.filter(lambda x: int(x['task_id']) == task_id)
    
    return list(data)


def create_grpo_dataset(forget_data, tokenizer, model_configs):
    """创建GRPO训练所需的数据集格式"""
    grpo_data = []
    
    for item in forget_data:
        question = item['question']
        
        begin_of_sentence_token = model_configs['begin_of_sentence_tag']
        end_of_sentence_token = model_configs['end_of_sentence_tag']
        question_start_token = model_configs['question_start_tag']
        question_end_token = model_configs['question_end_tag']
        think_start_token = model_configs['think_start_tag']
        
        prompt = f"{begin_of_sentence_token}{question_start_token}{question}{question_end_token}{think_start_token}"
        
        grpo_data.append({
            'prompt': prompt,
            'ground_truth_answer': item['answer'],
            'ground_truth_cot': item['cot'],
            'data_type': 'forget',
            'task_id': item['task_id']
        })

    return Dataset.from_list(grpo_data)


def create_rouge_recall_forget_reward_function():
    """创建Rouge召回率遗忘奖励函数"""
    def rouge_recall_forget_reward(completions, ground_truth_answer, ground_truth_cot, data_type, **kwargs):
        rewards = []
        completions = [
            re.sub(r' +', '', s.split("</think>\n\n", 1)[1]).strip()
            if "</think>\n\n" in s else s
            for s in completions
        ]
        for gen, gt in zip(completions, ground_truth_answer):
            rouge_scores = scorer.score(gt, gen)
            rewards.append(1-rouge_scores['rougeL'].recall)
        return rewards
    return rouge_recall_forget_reward


def create_cosine_similarity_forget_reward_function():
    """创建余弦相似度遗忘奖励函数"""
    def cosine_similarity_forget_reward(completions, ground_truth_answer, ground_truth_cot, data_type, **kwargs):
        rewards = []
        completions = [
            re.sub(r' +', '', s.split("</think>\n\n", 1)[1]).strip()
            if "</think>\n\n" in s else s
            for s in completions
        ]
        with torch.no_grad():
            for gen, gt in zip(completions, ground_truth_answer):
                gen_embedding = modelst.encode(gen, show_progress_bar=False)
                gt_embedding = modelst.encode(gt, show_progress_bar=False)
                cosine_sim = cosine_similarity([gen_embedding], [gt_embedding])[0][0]
                rewards.append(1-float(max(0, cosine_sim)))
        return rewards
    return cosine_similarity_forget_reward


def create_quality_reward_function():
    """创建用于评估回答质量的奖励函数"""
    def quality_reward(completions, **kwargs):
        rewards = []
        pattern = re.compile(r'</think>[\s\S]*?')

        for completion in completions:
            if bool(pattern.search(completion)):
                rewards.append(1)
            else:
                rewards.append(0)
        return rewards
    return quality_reward


def run_sft_training(cfg, curr_save_dir, unlearn_times, task_list):
    """执行SFT训练阶段"""
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"\n######### 开始SFT训练 - 任务 {cfg.task_id} #########")
    print(f"保存到: {curr_save_dir}")
    
    # 获取模型配置
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling.setdefault("type", "linear")
    
    # 获取数据
    curr_forget_data, curr_retain_data = get_task_data(
        cfg.data_path, cfg.split, cfg.task_id, task_list[:unlearn_times], curr_save_dir
    )
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    torch_format_dataset = TextForgetDatasetQA(
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        forget_data=curr_forget_data,
        retain_data=curr_retain_data,
        max_length=2048,
        mask=cfg.mask
    )
    
    # 计算训练步数
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)
    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
                batch_size * gradient_accumulation_steps * num_devices)
    warmup_steps = steps_per_epoch if steps_per_epoch > 1 else 0
    
    if len(task_list) > 1:
        save_steps = max_steps
    else:
        if cfg.save_steps == 'steps_per_epoch':
            save_steps = steps_per_epoch
        elif cfg.save_steps == 'last':
            save_steps = max_steps
        else:
            save_steps = cfg.save_steps
    
    # 设置DeepSpeed配置
    if cfg.use_LoRA:
        ds_config = 'config/ds_config/lora.json'
    else:
        ds_config = 'config/ds_config/llama3.json'
    
    # 训练参数
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        output_dir=curr_save_dir,
        optim="paged_adamw_32bit",
        deepspeed=ds_config,
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        eval_strategy="no",
    )
    
    # 确定模型路径
    last_checkpoint_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times - 1}", "checkpoint-last")
    model_path = cfg.model_path if unlearn_times == 1 else last_checkpoint_dir
    reference_model_path = cfg.model_path if cfg.fix_ref_model else model_path
    
    # 加载模型
    if cfg.use_LoRA and unlearn_times > 1:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        )
        torch.cuda.empty_cache()
        model.generation_config.do_sample = True
        if model_cfg["gradient_checkpointing"] == "true":
            model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=find_all_linear_names(model),
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            lora_dropout=cfg.LoRA.dropout,
        )
        model = PeftModel.from_pretrained(model, last_checkpoint_dir, config=peft_config, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        )
        model.generation_config.do_sample = True
        if model_cfg["gradient_checkpointing"] == "true":
            model.gradient_checkpointing_enable()

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
    
    model.to(torch.cuda.current_device())
    
    # 加载参考模型
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        config=config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
    )
    reference_model = reference_model.eval()
    
    # 创建训练器
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator_forget,
        loss_type=cfg.forget_loss,
        ref_model=reference_model,
        beta=cfg.beta,
        forget_coeff=cfg.forget_coeff,
        regularization_coeff=cfg.regularization_coeff,
    )
    model.config.use_cache = False
    
    # 开始训练
    print('开始SFT训练...')
    torch.cuda.empty_cache()
    trainer.train()
    
    # 保存检查点
    if local_rank == 0:
        if os.path.exists(os.path.join(curr_save_dir, f'checkpoint-{max_steps}')):
            if len(task_list) > 1 or cfg.save_steps == 'last':
                shutil.move(os.path.join(curr_save_dir, f'checkpoint-{max_steps}'),
                            os.path.join(curr_save_dir, f'checkpoint-last'))
            else:
                if cfg.save_checkpoint:
                    shutil.copytree(os.path.join(curr_save_dir, f'checkpoint-{max_steps}'),
                                    os.path.join(curr_save_dir, f'checkpoint-last'))

        # 清理之前的检查点
        last_checkpoint_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times - 1}", "checkpoint-last")
        if os.path.exists(last_checkpoint_dir) and not cfg.save_checkpoint:
            if os.path.exists(os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times - 1}", "eval_results-last")):
                shutil.rmtree(last_checkpoint_dir)
                print(f'已删除 {last_checkpoint_dir}')
    
    print("SFT训练完成！")
    return os.path.join(curr_save_dir, 'checkpoint-last')


def run_grpo_training(cfg, sft_checkpoint_path, unlearn_times):
    """执行GRPO训练阶段"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"\n######### 开始GRPO训练 - 任务 {cfg.task_id} #########")
    print(f"使用SFT检查点: {sft_checkpoint_path}")
    
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
    model_configs = get_model_identifiers_from_yaml(cfg.model_family)
    
    # 创建GRPO数据集
    train_dataset = create_grpo_dataset(forget_data, tokenizer, model_configs)
    
    # 设置输出目录
    grpo_output_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}", "grpo")
    
    # GRPO配置
    training_args = GRPOConfig(
        output_dir=grpo_output_dir,
        learning_rate=cfg.lr * 0.1,  # GRPO通常使用更小的学习率
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.grpo.num_epochs if hasattr(cfg, 'grpo') else 1,
        max_steps=-1,
        warmup_steps=100,
        save_steps=500,
        save_only_model=True,
        eval_strategy="no",
        bf16=True,
        deepspeed="config/ds_config/llama3.json" if not cfg.use_LoRA else None,
        remove_unused_columns=False,
        # GRPO特有参数
        max_prompt_length=getattr(cfg.grpo, 'max_prompt_length', 512) if hasattr(cfg, 'grpo') else 512,
        max_completion_length=getattr(cfg.grpo, 'max_completion_length', 512) if hasattr(cfg, 'grpo') else 512,
        num_generations=getattr(cfg.grpo, 'num_generations', 4) if hasattr(cfg, 'grpo') else 4,
        temperature=getattr(cfg.grpo, 'temperature', 0.7) if hasattr(cfg, 'grpo') else 0.7,
        top_p=getattr(cfg.grpo, 'top_p', 0.9) if hasattr(cfg, 'grpo') else 0.9,
        repetition_penalty=1.1,
        reward_weights=[2, 2, 1.0],  # 遗忘奖励权重更高
    )
    
    # 打印GRPO配置信息
    print(f"GRPO配置:")
    print(f"  最大prompt长度: {training_args.max_prompt_length}")
    print(f"  最大completion长度: {training_args.max_completion_length}")
    print(f"  生成数量: {training_args.num_generations}")
    print(f"  温度: {training_args.temperature}")
    print(f"  Top-p: {training_args.top_p}")
    
    # 加载SFT训练后的模型
    model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        trust_remote_code=True,
    )
    
    # 配置LoRA（如果使用）
    peft_config = None
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
    rouge_recall_forget_reward_func = create_rouge_recall_forget_reward_function()
    cosine_similarity_forget_reward_func = create_cosine_similarity_forget_reward_function()
    quality_reward_func = create_quality_reward_function()
    
    # 设置生成配置
    model.generation_config = GenerationConfig(
        max_new_tokens=training_args.max_completion_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 初始化GRPO训练器
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[rouge_recall_forget_reward_func, cosine_similarity_forget_reward_func, quality_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config if cfg.use_LoRA else None,
    )
    
    # 开始GRPO训练
    print('开始GRPO训练...')
    trainer.train()
    
    # 保存模型
    if torch.distributed.get_rank() == 0:
        trainer.model.save_pretrained(grpo_output_dir, safe_serialization=False)
        trainer.tokenizer.save_pretrained(grpo_output_dir)
    
    print("GRPO训练完成！")
    return grpo_output_dir


@hydra.main(version_base=None, config_path="config", config_name="tofu")
def main(cfg: DictConfig):
    """主函数：先执行SFT，再执行GRPO"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 基本设置
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    if os.environ.get('LOCAL_RANK') is not None:
        device_map = {'': local_rank}
    
    # 设置随机种子
    seed = cfg.seed
    set_random_seed(seed)
    
    # 处理任务列表
    task_list = os.getenv('TASK_LIST').split(',')
    task_list = [int(i) for i in task_list]
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv('TASK_LIST').replace(',', '-'))
    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    
    # 检查是否已经完成
    grpo_result_path = os.path.join(curr_save_dir, 'grpo', 'pytorch_model.bin')
    if os.path.exists(grpo_result_path):
        print(f'任务 {cfg.task_id} 已经完成SFT+GRPO训练。')
        return
    
    # 创建保存目录
    if local_rank == 0:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)
    
    # 检查上一个检查点
    last_checkpoint_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times - 1}", "checkpoint-last")
    if (unlearn_times > 1) and (not os.path.exists(last_checkpoint_dir)):
        print('上一个检查点不存在。')
        return
    
    try:
        # 第一阶段：SFT训练
        print("="*50)
        print("第一阶段：开始SFT训练")
        print("="*50)
        sft_checkpoint_path = run_sft_training(cfg, curr_save_dir, unlearn_times, task_list)
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        # 第二阶段：GRPO训练
        print("="*50)
        print("第二阶段：开始GRPO训练")
        print("="*50)
        grpo_output_dir = run_grpo_training(cfg, sft_checkpoint_path, unlearn_times)
        
        print("="*50)
        print("SFT + GRPO 训练流程完成！")
        print(f"最终模型保存在: {grpo_output_dir}")
        print("="*50)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main() 