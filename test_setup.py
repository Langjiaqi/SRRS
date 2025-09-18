#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置验证脚本
检查环境和路径是否配置正确
"""

import os
import sys

def check_paths():
    """检查路径配置"""
    print("=== 路径检查 ===")
    
    # 检查配置文件
    try:
        from test_config import MODEL_PATH, DATA_PATH, OUTPUT_DIR
        print(f"✓ 配置文件加载成功")
        print(f"模型路径: {MODEL_PATH}")
        print(f"数据路径: {DATA_PATH}")
        print(f"输出路径: {OUTPUT_DIR}")
    except ImportError:
        print("! 警告: test_config.py 不存在，将使用默认配置")
        MODEL_PATH = "/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch1_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last"
        DATA_PATH = "/ljq/rtofu/results/rtofu/llama3-8b/forget01/GA1/seed_1001/epoch1_1e-05_FixRefFalse_maskTrue_1.0_1.0/1/unlearn_times_1/task_data"
        OUTPUT_DIR = "/ljq/rtofu/test_results"
    
    # 检查模型路径
    if os.path.exists(MODEL_PATH):
        print(f"✓ 模型路径存在: {MODEL_PATH}")
    else:
        print(f"✗ 模型路径不存在: {MODEL_PATH}")
    
    # 检查数据路径
    if os.path.exists(DATA_PATH):
        print(f"✓ 数据路径存在: {DATA_PATH}")
        
        # 检查数据文件
        data_files = [
            "forget_perturbed.json",
            "retain_perturbed.json", 
            "real_authors_perturbed.json",
            "world_facts_perturbed.json"
        ]
        
        found_files = []
        for filename in data_files:
            filepath = os.path.join(DATA_PATH, filename)
            if os.path.exists(filepath):
                found_files.append(filename)
                print(f"  ✓ 找到数据文件: {filename}")
        
        if not found_files:
            print("  ! 警告: 未找到标准数据文件，将使用示例数据")
        else:
            print(f"  ✓ 共找到 {len(found_files)} 个数据文件")
    else:
        print(f"✗ 数据路径不存在: {DATA_PATH}")
    
    # 检查输出目录
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"✓ 输出目录已准备: {OUTPUT_DIR}")
    except Exception as e:
        print(f"✗ 无法创建输出目录: {e}")

def check_dependencies():
    """检查依赖"""
    print("\n=== 依赖检查 ===")
    
    required_modules = [
        'torch',
        'transformers', 
        'rouge_score',
        'metrics',
        'utils'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")

def check_gpu():
    """检查GPU"""
    print("\n=== GPU检查 ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    显存: {torch.cuda.get_device_properties(i).total_memory // 1024**3} GB")
        else:
            print("! 警告: GPU不可用，将使用CPU（速度会很慢）")
    except ImportError:
        print("✗ 无法导入torch")

def main():
    """主函数"""
    print("模型测试环境检查")
    print("=" * 50)
    
    check_paths()
    check_dependencies() 
    check_gpu()
    
    print("\n" + "=" * 50)
    print("检查完成！")
    print("\n如果所有检查都通过，可以运行:")
    print("  python testmodel.py")
    print("或者:")
    print("  python run_test_multi.py")

if __name__ == "__main__":
    main() 