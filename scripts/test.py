#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 模型测试脚本

此脚本用于测试训练好的 YOLO 模型，主要功能包括：
- 加载训练好的模型权重
- 对输入图像进行目标检测
- 输出检测结果，包括：
  - 边界框坐标（多种格式）
  - 类别名称
  - 置信度分数

支持的边界框格式：
- xywh: 中心点坐标和宽高
- xywhn: 归一化的中心点坐标和宽高
- xyxy: 左上角和右下角坐标
- xyxyn: 归一化的左上角和右下角坐标

作者: Jerry
创建日期: 2024-03-21
"""

from ultralytics import YOLO

# Load a model
model = YOLO("/home/yjr/下载/qianrushi/models/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box