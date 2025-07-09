#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 数据集配置文件生成脚本

此脚本用于自动生成 YOLO 训练所需的 dataset.yaml 配置文件，主要功能包括：
- 读取数据集根目录下的 classes.txt 文件获取类别信息
- 自动检测训练集、验证集和测试集目录
- 生成符合 YOLO 格式的 YAML 配置文件

配置文件结构：
- train: 训练集图片目录路径
- val: 验证集图片目录路径
- test: 测试集图片目录路径（可选）
- nc: 类别数量
- names: 类别名称列表

作者: Jerry
创建日期: 2024-03-21
"""

import os
import yaml

def generate_yaml(dataset_root, output_path="dataset.yaml"):
    images_dir = os.path.join(dataset_root, "images")
    classes_file = os.path.join(dataset_root, "classes.txt")

    if not os.path.exists(classes_file):
        print("找不到 classes.txt，请确保它存在于数据集根目录！")
        return

    # 读取类别名
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    num_classes = len(class_names)

    # 构造yaml结构
    data = {
        "train": os.path.abspath(os.path.join(images_dir, "train")),
        "val": os.path.abspath(os.path.join(images_dir, "val")),
        "nc": num_classes,
        "names": class_names
    }

    # 如果存在 test 目录，则加上
    test_dir = os.path.join(images_dir, "test")
    if os.path.exists(test_dir):
        data["test"] = os.path.abspath(test_dir)

    # 保存为yaml文件
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"✅ YAML 文件已生成：{output_path}")

# 使用示例
generate_yaml("/home/yjr/下载/qianrushi/dataset")