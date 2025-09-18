#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的测试启动脚本
直接运行指定路径的模型和数据进行多输出测试
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append('/ljq/rtofu')

from testmodel import main

if __name__ == "__main__":
    print("启动模型多输出测试...")
    try:
        main()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 