#!/usr/bin/env python3
"""
启动TensorBoard来查看GRPO训练日志的便捷脚本
"""

import os
import sys
import subprocess
import argparse
import glob

def find_tensorboard_logs(base_dir):
    """查找所有TensorBoard日志目录"""
    log_dirs = []
    
    # 查找grpo_seed_*目录下的tensorboard_logs
    grpo_dirs = glob.glob(os.path.join(base_dir, "grpo_seed_*"))
    
    for grpo_dir in grpo_dirs:
        # 查找unlearn_times_*目录
        unlearn_dirs = glob.glob(os.path.join(grpo_dir, "unlearn_times_*"))
        for unlearn_dir in unlearn_dirs:
            tb_dir = os.path.join(unlearn_dir, "tensorboard_logs")
            if os.path.exists(tb_dir):
                log_dirs.append(tb_dir)
    
    return log_dirs

def main():
    parser = argparse.ArgumentParser(description="启动TensorBoard查看GRPO训练日志")
    parser.add_argument("--logdir", type=str, help="指定TensorBoard日志目录")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard端口号 (默认: 6006)")
    parser.add_argument("--host", type=str, default="localhost", help="TensorBoard主机地址 (默认: localhost)")
    parser.add_argument("--base_dir", type=str, default=".", help="搜索日志的基础目录 (默认: 当前目录)")
    
    args = parser.parse_args()
    
    if args.logdir:
        # 用户指定了日志目录
        if not os.path.exists(args.logdir):
            print(f"错误: 指定的日志目录不存在: {args.logdir}")
            return 1
        logdir = args.logdir
    else:
        # 自动查找日志目录
        log_dirs = find_tensorboard_logs(args.base_dir)
        
        if not log_dirs:
            print(f"在 {args.base_dir} 中未找到TensorBoard日志目录")
            print("请确保已运行GRPO训练且启用了TensorBoard记录")
            return 1
        elif len(log_dirs) == 1:
            logdir = log_dirs[0]
            print(f"找到日志目录: {logdir}")
        else:
            print("找到多个日志目录:")
            for i, dir_path in enumerate(log_dirs):
                print(f"  {i+1}. {dir_path}")
            
            while True:
                try:
                    choice = input(f"请选择要查看的日志目录 (1-{len(log_dirs)}) 或按Enter查看所有: ").strip()
                    if not choice:
                        # 查看所有日志
                        logdir = args.base_dir
                        break
                    else:
                        choice = int(choice) - 1
                        if 0 <= choice < len(log_dirs):
                            logdir = log_dirs[choice]
                            break
                        else:
                            print(f"请输入1到{len(log_dirs)}之间的数字")
                except (ValueError, KeyboardInterrupt):
                    print("\n已取消")
                    return 1
    
    print(f"启动TensorBoard...")
    print(f"日志目录: {logdir}")
    print(f"访问地址: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止TensorBoard")
    
    try:
        # 启动TensorBoard
        cmd = [
            "tensorboard",
            "--logdir", logdir,
            "--port", str(args.port),
            "--host", args.host,
            "--reload_interval", "30"  # 每30秒重新加载日志
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nTensorBoard已停止")
    except FileNotFoundError:
        print("错误: 未找到tensorboard命令")
        print("请安装tensorboard: pip install tensorboard")
        return 1
    except Exception as e:
        print(f"启动TensorBoard时出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 