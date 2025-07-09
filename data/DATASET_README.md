# 🍅 Tomato Disease Detection Dataset

本数据集用于训练 YOLO 模型，目标是识别番茄叶片的健康状态与多种病害，适用于病虫害检测项目。

---

## 📦 数据内容

- **图像格式**：.jpg/.png
- **标签格式**：YOLO 格式（.txt 与图像同名）
- **类别标签（共 9 类）**：

```
healthy  
mosaic_virus  
early_blight  
late_blight  
septoria  
yellow_leaf_curl_virus  
leaf_mold  
leaf_miner  
spider_mites
```

---

## 🔗 数据集下载链接（Google Drive）

👉 [点击下载数据集](https://drive.google.com/drive/folders/1zkkdd96aSNLfui1OQENY5ynQTnDdi8kE?usp=drive_link)

---

## 📂 下载后建议目录结构

```
your_project/
├── data/
│   ├── raw/                 # 原始图片与标签
│   │   ├── images/
│   │   └── labels/
│   └── DATASET_README.md
```

---

## 🛠 如何使用

1. 下载并解压数据集到 `data/raw/`
2. 确保 `dataset.yaml` 配置文件路径正确
3. 启动 YOLO11n 训练脚本进行模型训练

---

## 📜 使用须知

- 本数据仅供学术研究与模型训练使用
- 若使用公开模型请注明数据来源
