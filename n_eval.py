import csv
import json
import os
import shutil
import warnings
import vllm

import hydra
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from metrics.n_metrics import get_all_evals, get_dataloader, get_eval_results
from utils import get_model_identifiers_from_yaml

warnings.filterwarnings('ignore')


def model_eval(cfg, task_id, unlearn_times, model, tokenizer, save_dir, curr_forget_path, eval_unlearn_step=None,
               num_rollouts=3, temperature=0.7, use_vllm=True):
    """
    模型评估函数，支持多回复生成
    Args:
        num_rollouts: 每个prompt生成的回复数量
        temperature: 采样温度
        use_vllm: 是否使用vLLM
    """
    eval_unlearn_step = 'last' if eval_unlearn_step == None else eval_unlearn_step
    aggregated_eval_logs = {}
    
    # 添加生成参数到配置中
    if not hasattr(cfg.eval, 'num_rollouts'):
        cfg.eval.num_rollouts = num_rollouts
    if not hasattr(cfg.eval, 'temperature'):
        cfg.eval.temperature = temperature
    if not hasattr(cfg.eval, 'use_vllm'):
        cfg.eval.use_vllm = use_vllm
        
    # 如果使用vLLM，添加模型路径到配置中
    if use_vllm and not hasattr(cfg.eval, 'model_path'):
        cfg.eval.model_path = cfg.model_path
    
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(
            zip(cfg.eval.data_path, cfg.eval.split_list, cfg.eval.question_key, cfg.eval.answer_key, cfg.eval.eval_task,
                cfg.eval.base_answer_key, cfg.eval.perturbed_answer_key)):
        if eval_task != 'eval_log_forget':
            continue
        if eval_task == 'eval_log_forget':
            folder = curr_forget_path
            split = "forget_perturbed"

        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f"{eval_task}_multi_{num_rollouts}_{temperature}.json")

        if os.path.exists(save_filename):
            print(f"跳过 {eval_task}，因为 {save_filename} 已存在")
            eval_logs = json.load(open(save_filename, 'r'))
        else:
            print(f"开始评估 {eval_task}，使用 {num_rollouts} 个rollouts，温度 {temperature}")
            eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
                cfg.eval, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key,
                perturbed_answer_key)

            eval_logs = get_all_evals(cfg.eval, model, tokenizer, folder, split, eval_task, eval_dataloader,
                                      base_eval_dataloader, perturb_dataloader, True, 
                                      num_rollouts=num_rollouts, temperature=temperature, use_vllm=use_vllm)

            with open(save_filename, "w") as f:
                json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(
        save_dir, f"eval_log_aggregated_multi_{num_rollouts}_{temperature}.json")
    with open(aggregated_eval_log_filename, "w") as f:
        json.dump(aggregated_eval_logs, f, indent=4)

    eval_results = get_eval_results(aggregated_eval_logs)
    aggregate_stat = {**eval_results}

    print("=== 多回复生成评估结果 ===")
    print(f"Rollouts: {num_rollouts}, Temperature: {temperature}")
    for key, value in aggregate_stat.items():
        print(f"{key}: {value}")

    aggregate_stat['split'] = cfg.split
    aggregate_stat['forget_loss'] = cfg.forget_loss
    aggregate_stat['forget_coeff'] = cfg.forget_coeff
    aggregate_stat['regularization_coeff'] = cfg.regularization_coeff
    aggregate_stat['learning_rate'] = cfg.lr
    aggregate_stat['epochs'] = cfg.num_epochs
    aggregate_stat['fix_ref_model'] = cfg.fix_ref_model
    aggregate_stat['mask'] = cfg.mask
    aggregate_stat['unlearn_step'] = eval_unlearn_step
    aggregate_stat['task_id'] = task_id
    aggregate_stat['unlearn_times'] = unlearn_times
    aggregate_stat['num_rollouts'] = num_rollouts
    aggregate_stat['temperature'] = temperature
    aggregate_stat['use_vllm'] = use_vllm

    # 保存结果到文本文件
    results_filename = f"unlearning_results_multi_{num_rollouts}_{temperature}.txt"
    with open(os.path.join(save_dir, results_filename), 'w') as txtfile:
        txtfile.write(f"=== 多回复生成评估结果 ===\n")
        txtfile.write(f"Rollouts: {num_rollouts}, Temperature: {temperature}\n")
        txtfile.write(f"Use vLLM: {use_vllm}\n\n")
        for key, value in aggregate_stat.items():
            txtfile.write(f"{key}: {value}\n")

    # 保存结果到CSV文件
    csv_filename = f"unlearning_results_multi_{num_rollouts}_{temperature}.csv"
    save_file = os.path.join(save_dir, csv_filename)
    with open(save_file, 'w') as f:
        w = csv.DictWriter(f, aggregate_stat.keys())
        w.writeheader()
        w.writerow(aggregate_stat)

    # 保存到总的结果文件中
    all_task_save_file = os.path.join(cfg.save_dir, "all_unlearning_results_multi.csv")
    if not os.path.exists(all_task_save_file) or os.path.getsize(all_task_save_file) == 0:
        with open(all_task_save_file, 'w') as f:
            w = csv.DictWriter(f, aggregate_stat.keys())
            w.writeheader()
            w.writerow(aggregate_stat)
    else:
        with open(all_task_save_file, 'a') as f:
            w = csv.DictWriter(f, aggregate_stat.keys())
            w.writerow(aggregate_stat)

    return eval_results


@hydra.main(version_base=None, config_path="config", config_name="n_eval")
def main(cfg):
    # 从Hydra配置获取参数，不再使用环境变量
    num_rollouts = cfg.eval.num_rollouts
    temperature = cfg.eval.temperature
    use_vllm = cfg.eval.use_vllm
    
    print(f"评估参数: rollouts={num_rollouts}, temperature={temperature}, use_vllm={use_vllm}")
    print(f"模型路径: {cfg.model_path}")
    
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    else:
        device_map = "auto"

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    # 从配置文件获取task_list，如果没有则使用单个task_id
    task_list = getattr(cfg.experiment, 'task_list', [cfg.task_id]) if hasattr(cfg, 'experiment') else [cfg.task_id]
    task_list_str = ','.join(map(str, task_list))
    cfg.save_dir = os.path.join(cfg.save_dir, task_list_str.replace(',', '-'))

    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_data_path = os.path.join(curr_save_dir, "task_data")
    curr_forget_perturbed_path = os.path.join(os.path.dirname(cfg.model_path), "task_data")

    curr_checkpoint_dir = os.path.join(curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}")
    if cfg.eval_unlearn_step == 0:
        curr_checkpoint_dir = cfg.model_path
    elif cfg.eval_unlearn_step == 1:            ##base model
        curr_checkpoint_dir = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    elif cfg.eval_unlearn_step == "last":        ##use the exact model_path for last checkpoint
        curr_checkpoint_dir = cfg.model_path
    else:
        if not os.path.exists(curr_checkpoint_dir):
            print(f'{curr_checkpoint_dir} does not exist.')
            exit()

    curr_eval_dir = os.path.join(curr_save_dir, f'eval_results-{cfg.eval_unlearn_step}-multi')
    
    # 检查是否已经评估过
    expected_csv = os.path.join(curr_eval_dir, f'unlearning_results_multi_{num_rollouts}_{temperature}.csv')
    if os.path.exists(expected_csv):
        print(f'{expected_csv} 已经评估过了.')
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling.setdefault("type", "linear")
    
    # 如果不使用vLLM，加载transformers模型
    if not use_vllm:
        if cfg.use_LoRA:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                config=config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
            model = PeftModel.from_pretrained(model, curr_checkpoint_dir)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                curr_checkpoint_dir,
                config=config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
        model = model.eval()
    else:
        # 使用vLLM时，传入None作为模型（在n_metrics中会初始化vLLM）
        model = None
        # 但仍需要加载用于损失计算的模型
        if cfg.use_LoRA:
            loss_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                config=config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
            loss_model = PeftModel.from_pretrained(loss_model, curr_checkpoint_dir)
            loss_model = loss_model.merge_and_unload()
        else:
            loss_model = AutoModelForCausalLM.from_pretrained(
                curr_checkpoint_dir,
                config=config,
                attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
        model = loss_model.eval()
        
        # 添加模型路径到配置中，供vLLM使用
        cfg.model_path = curr_checkpoint_dir

    eval_results = model_eval(cfg, cfg.task_id, unlearn_times, model, tokenizer, curr_eval_dir, curr_forget_perturbed_path,
                              cfg.eval_unlearn_step, num_rollouts=num_rollouts, temperature=temperature, use_vllm=use_vllm)

    print("评估完成!")


if __name__ == "__main__":
    main() 