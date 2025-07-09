#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 目标检测模型训练脚本（方案2：动态加载自定义 hyper-params + 单 GPU + 关闭 AMP & Fast Image Access）

此脚本示例：
- 加载预训练模型(YOLO11n)
- 动态读取 `hyp_custom.yaml` 中所有增强和超参
- 强制使用单 GPU，避免 DDP 错误
- 关闭 AMP 检查和 Fast Image Access，避免 OpenCV 相关错误

作者: Jerry
创建日期: 2024-03-21
"""

import os
import yaml
import torch
import numpy as np
from ultralytics import YOLO


def main():
    # 文件路径（请根据实际情况修改）
    model_path   = "/home/yjr/下载/qianrushi/pretrained/yolo11n.pt"
    dataset_yaml = "/home/yjr/下载/qianrushi/dataset/dataset.yaml"
    hyp_yaml     = "/home/yjr/下载/qianrushi/scripts/hyp_custom.yaml"

    # 校验文件存在性
    for path in (model_path, dataset_yaml, hyp_yaml):
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

    # 读取超参与增强配置
    with open(hyp_yaml, 'r', encoding='utf-8') as f:
        hyp = yaml.safe_load(f)
    print(f"Loaded hyperparameters from {hyp_yaml}")

    # 设置随机种子，保证可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 强制单 GPU 训练
    device_str = "0"  # 如果需要 CPU，改为 "cpu"
    print(f"Using device: {device_str}")

    # 加载模型
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully.")

    # 开始训练
    print("Starting training...")
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=48,
        device=device_str,
        project="runs/train",
        name="exp_hyp",
        save=True,
        save_period=10,
        optimizer="SGD",
        amp=False,      # 关闭自动混合精度检查
        fast=False,     # 关闭 Fast image access，使用常规 cv2.imread
        cache=False,    # 关闭缓存加速，确保使用 cv2.imread
        **hyp          # 展开自定义超参和增强配置
    )
    print("Training completed.")

    # 验证性能
    print("Validating model...")
    metrics = model.val()
    print("Validation results:", metrics)

    # 导出为 ONNX
    print("Exporting to ONNX format...")
    export_path = model.export(format="onnx")
    print(f"Export completed. File saved at {export_path}")


if __name__ == "__main__":
    main()
