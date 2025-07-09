#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CUDA 环境检测脚本

此脚本用于检测当前环境的 CUDA 和 cuDNN 状态，包括：
- CUDA 是否可用
- cuDNN 是否可用
- CUDA 版本号
- cuDNN 版本号

作者: Jerry
创建日期: 2024-03-21
"""

import torch
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())