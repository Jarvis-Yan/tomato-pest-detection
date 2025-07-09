# 番茄病虫害识别系统 —— 图像推理子系统 (ELF2 GW3588)

本文件仅覆盖 **图像识别后端**（NPU + YOLOv8）的使用方法。  
传感器 / STM32 与前端页面请见 `docs/` 目录或相应仓库。

---

## 目录结构

```text
├── config/                  # 统一的 YAML / INI 参数（可选）
├── data/                    # 小批量测试数据；非必须
├── dataset/                 # 训练数据示例（压缩或精简版）
├── models/
│   ├── best.onnx            # 训练导出的 ONNX（备份）
│   ├── best.pt              # Ultralytics‑YOLOv8 权重
│   └── yolov8.rknn          # **⇦ 推理用，放置于板卡 /mnt/tfcard/...**
├── pretrained/              # 预训练 YOLOv8n / YOLOv11 等
├── src/                     # 运行入口均在这里
│   ├── video.py             # 纯 OpenCV 测试摄像头 / RTSP（无 NPU）
│   ├── yolov8.py            # RKNN‑Lite2 推理封装
│   ├── stream_processor.py  # 边推理边上传结果到 Django 服务器
│   └── synset_label.py      # 读取 `classes.txt` → id ↔ name
├── docs/
│   └── assets/              # README 静态资源（GIF / PNG 等）
│       ├── demo_vehicle.gif     # ① 小车移动 + 摄像头
│       ├── demo_inference.gif   # ② 实时识别过程
│       └── demo_board.gif       # ③ 板端显示效果
└── classes.txt              # 类别文本（与训练一致，一行一个）
```

---

## 1. 环境准备（一次性）

💾 **TF‑Card** 建议 *ext4*，挂载到 `/mnt/tfcard` 并软链至 `~/tfcard`。  
推荐将所有模型及 Conda 环境安装在此卡，避免板载 eMMC 空间不足。

### 1.1 RKNN‑Toolkit‑Lite2（板卡端）

```bash
# 假设 wheel 位于 ~/tfcard/rknn_toolkit_lite2‑2.1.0‑cp310‑*.whl
pip install --no-cache-dir ~/tfcard/rknn_toolkit_lite2-2.1.0-cp310-*.whl
```

> 若报 “not a supported wheel”，请确认 **Python 3.10 / aarch64** 与文件后缀  
> `cp310‑linux_aarch64.whl` 完全匹配。

### 1.2 Conda（可选）

```bash
# 将 Miniforge 安装到 ~/tfcard/conda
bash Miniforge3-Linux-aarch64.sh -p ~/tfcard/conda
source ~/tfcard/conda/etc/profile.d/conda.sh
conda create -p ~/tfcard/conda/envs/tomato python=3.10 -y
conda activate ~/tfcard/conda/envs/tomato
pip install -r requirements.txt
```

### 1.3 环境变量

```bash
export PYTHONPATH=/mnt/tfcard:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH   # 默认已包含
```

---

## 2. 快速运行

### 2.1 本地摄像头 / RTSP 测试（窗口显示）

```bash
cd src
python video.py rtsp://<CAM_IP>:554/11   # 仅 CPU 解码，不占用 NPU
```

### 2.2 NPU + 推理 + 上传

```bash
cd src
python stream_processor.py \
    --model  ../models/yolov8.rknn \
    --rtsp   rtsp://<CAM_IP>:554/11 \
    --server http://<DJANGO_IP>:8000/ \
    [--nogui]
```

| 参数      | 说明                                               |
|-----------|----------------------------------------------------|
| `--model` | RKNN 模型路径；默认为 `../models/yolov8.rknn`      |
| `--rtsp`  | 摄像头地址，缺省则读取本地 **0 号** 设备           |
| `--server`| Django API 根地址，用于 `POST` JSON 结果           |
| `--nogui` | 加上后不弹出 OpenCV 窗口（headless 模式）          |

---

## 🚀 Demo

<table>
  <tr>
    <td align="center">
      <strong>机体巡航</strong><br>
      <img src="docs/assets/demo_vehicle.gif" width="260">
    </td>
    <td align="center">
      <strong>实时识别</strong><br>
      <img src="docs/assets/demo_inference.gif" width="260">
    </td>
    <td align="center">
      <strong>板端显示</strong><br>
      <img src="docs/assets/demo_board.gif" width="260">
    </td>
  </tr>
</table>

> 所有 GIF 均已压缩至 **&lt; 10 MB**，放置在 `docs/assets/`，确保在线 / 离线均可预览。

---

## 3. 常见问题

| 现象 / 日志片段                                     | 原因 & 解决方案                                                                           |
|-----------------------------------------------------|------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: rknnlite`                     | wheel 未安装或 `PYTHONPATH` 未指向 TF‑Card 的 *site‑packages*                             |
| `Unsupport data format: NHWC / NCHW`                | 模型导出期望的数据布局与 `yolov8.py` 预处理不符；检查 `np.transpose` 或重新导出模型       |
| `RKNN Model version 2.3.0 not match runtime 2.1.0`  | SDK 版本过低；就地降级模型或升级板卡中的 `librknnrt.so`                                   |
| `OpenCV Can't initialize GTK backend`               | `DISPLAY` 为空（SSH）或 Wayland；运行前 `export DISPLAY=:0` 或在命令行加 `--nogui`         |

---

## 4. 模型再导出（ONNX → RKNN）

完整脚本见 `models/convert.py`，核心步骤如下：

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(
    target_platform='rk3588',
    mean_values=[[123.675, 116.28, 103.53]],
    std_values=[[58.395, 57.12, 57.375]],
    quantize_input_node=True,
)

rknn.load_onnx(model='best.onnx')
rknn.build(do_quantization=True, dataset='dataset/calib.txt')
rknn.export_rknn('yolov8.rknn')
```

---

## 5. 参考 / 鸣谢

- Rockchip **RKNN‑Toolkit‑Lite2 v2.1.0**
- **Ultralytics YOLOv8**
- [ELFBoard 官方教程](https://www.elfboard.com/information/detail.html?id=7)

---

如发现与 `src` 目录启动流程不符的步骤，请在 **Issue** / 团队群内反馈 👋