import cv2
import numpy as np
import onnxruntime
from rknn.api import RKNN

# 模型路径
onnx_path = './best.onnx'
rknn_path = './best.rknn'
image_path = './test.jpg'

# 图像预处理函数（按模型需求调整）
def preprocess(img_path, size=(640, 640)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, [2, 0, 1])  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

# 预处理图像
input_data = preprocess(image_path)

# ONNX 推理
onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
onnx_output = onnx_session.run(None, {onnx_input_name: input_data})[0]
print("ONNX output shape:", onnx_output.shape)
print("ONNX output sample:", onnx_output.flatten()[:10])

# RKNN 推理
rknn = RKNN()
rknn.load_rknn(rknn_path)
rknn.init_runtime()
rknn_output = rknn.inference(inputs=[input_data])[0]
print("RKNN output shape:", np.array(rknn_output).shape)
print("RKNN output sample:", np.array(rknn_output).flatten()[:10])

rknn.release()