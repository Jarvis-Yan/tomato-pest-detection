"""
RKNN YOLOv8 推理与流处理主模块

核心功能：
- RKNN模型加载与推理
- RTSP流实时采集与多线程处理
- 目标检测后处理与可视化
- 兼容YOLO风格的帧处理接口

依赖：
- OpenCV (cv2)：视频与图像处理
- rknnlite：RKNN模型推理
- numpy：数值计算
- PIL：图像处理
- queue/threading：多线程与队列
"""
# ------------------- 导入必要的库 -------------------
import os  # 操作系统接口
import cv2  # OpenCV库，用于图像处理
from rknnlite.api import RKNNLite  # RKNN Lite推理库
import numpy as np  # 数值计算库
import stat  # 文件状态
from PIL import Image  # 图像处理库
import argparse  # 命令行参数解析
import time  # 时间相关功能
import threading  # 多线程支持
import queue  # 队列数据结构


# ------------------- 配置参数 -------------------
DEFAULT_RKNN_MODEL = "/home/elf/tfcard/yolov8_test/yolov8.rknn"  # 默认RKNN模型路径
DEFAULT_CLASSES = [  # 默认类别列表
    "healthy",
    "mosaic_virus",
    "early_blight",
    "late_blight",
    "septoria",
    "yellow_leaf_curl_virus",
    "leaf_mold",
    "leaf_miner",
    "spider_mites"
]
DEFAULT_COCO_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 默认COCO ID列表
OBJ_THRESH = 0.45  # 目标检测置信度阈值
NMS_THRESH = 0.45  # 非极大值抑制阈值
MODEL_SIZE = (640, 640)  # 模型输入尺寸

# ------------------- 工具函数 -------------------
def letter_box(im, new_shape, pad_color=(0, 0, 0)):
    """
    图像缩放和填充，保持宽高比不变
    :param im: 输入图像
    :param new_shape: 目标尺寸 (height, width)
    :param pad_color: 填充颜色
    :return: 处理后的图像
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return im

def filter_boxes(boxes, box_confidences, box_class_probs):
    """
    过滤检测框，基于置信度和类别概率
    :param boxes: 检测框坐标
    :param box_confidences: 检测框置信度
    :param box_class_probs: 检测框类别概率
    :return: 过滤后的boxes, classes和scores
    """
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """
    非极大值抑制(NMS)
    :param boxes: 检测框坐标
    :param scores: 检测框得分
    :return: 保留的检测框索引
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def softmax(x, axis=None):
    """
    softmax函数
    :param x: 输入数据
    :param axis: 计算softmax的轴
    :return: softmax计算结果
    """
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def dfl(position):
    """
    Distribution Focal Loss (DFL)处理
    :param position: 位置数据
    :return: 处理后的结果
    """
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
    """
    处理检测框位置信息，输出xyxy格式
    :param position: 位置数据
    :return: xyxy格式检测框
    """
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([MODEL_SIZE[1] // grid_h, MODEL_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def post_process(input_data):
    """
    模型输出后处理，返回检测框、类别、分数
    :param input_data: 模型输出数据
    :return: boxes, classes, scores
    """
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
    if not nclasses and not nscores:
        return None, None, None
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

def draw_detections(img, left, top, right, bottom, score, class_id, classes, color_palette):
    """
    在图像上绘制单个检测框和标签
    :param img: 输入图像
    :param left, top, right, bottom: 检测框坐标
    :param score: 置信度
    :param class_id: 类别ID
    :param classes: 类别名称列表
    :param color_palette: 颜色调色板
    """
    color = color_palette[class_id]
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), color, 2)
    label = f"{classes[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = left
    label_y = top - 10 if top - 10 > label_height else top + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw(image, boxes, scores, classes, class_names, color_palette):
    """
    在图像上绘制所有检测结果
    :param image: 输入图像
    :param boxes: 检测框列表
    :param scores: 分数列表
    :param classes: 类别列表
    :param class_names: 类别名称列表
    :param color_palette: 颜色调色板
    """
    img_h, img_w = image.shape[:2]
    x_factor = img_w / MODEL_SIZE[0]
    y_factor = img_h / MODEL_SIZE[1]
    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(_b) for _b in box]
        left = int(x1 * x_factor)
        top = int(y1 * y_factor)
        right = int(x2 * x_factor)
        bottom = int(y2 * y_factor)
        draw_detections(image, left, top, right, bottom, score, cl, class_names, color_palette)

# ------------------- RTSP处理类（带队列） -------------------
class RKNNRTSPProcessor:
    """
    RTSP流处理器，基于RKNN Lite进行推理和多线程采集
    """
    def __init__(self, model_path, class_names, coco_id_list, save_path=None, queue_size=10, queue_delay=0.02):
        """
        初始化
        :param model_path: RKNN模型路径
        :param class_names: 类别名称列表
        :param coco_id_list: COCO ID列表
        :param save_path: 输出视频路径
        :param queue_size: 帧队列大小
        :param queue_delay: 采集延迟
        """
        self.model_path = model_path
        self.class_names = class_names
        self.coco_id_list = coco_id_list
        self.save_path = save_path
        self.rknn_lite = RKNNLite()
        self.color_palette = np.random.uniform(0, 255, size=(len(class_names), 3))
        self.queue = queue.Queue(maxsize=queue_size)
        self.queue_delay = queue_delay
        self._init_model()
        self.stop_flag = False

    def _init_model(self):
        """加载并初始化RKNN模型"""
        print('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(self.model_path)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('done')
        print('--> Init runtime environment')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

    def _frame_reader(self, cap):
        """
        读取线程：不断从RTSP流读取帧，放入队列。队列满时丢弃新帧。
        :param cap: OpenCV视频捕获对象
        """
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                self.queue.put(frame, timeout=0.1)
            except queue.Full:
                pass
            time.sleep(self.queue_delay)

    def process_rtsp(self, rtsp_url, show_progress=True):
        """
        处理RTSP流，主推理循环
        :param rtsp_url: RTSP流地址
        :param show_progress: 是否显示进度
        :return: 统计信息字典
        """
        print(f"尝试打开RTSP流: {rtsp_url}")
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 RTSP 流（FFmpeg 后端）: {rtsp_url}")
        print("RTSP流打开成功")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = None
        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.save_path, fourcc, fps if fps > 0 else 25, (frame_width, frame_height))
        frame_times = []
        frame_idx = 0
        all_detections = {}
        self.stop_flag = False
        reader_thread = threading.Thread(target=self._frame_reader, args=(cap,))
        reader_thread.start()
        try:
            while True:
                try:
                    frame = self.queue.get(timeout=1)
                    if frame is None or frame.std() < 5 or frame.mean() > 250:
                        continue
                except queue.Empty:
                    if not reader_thread.is_alive():
                        break
                    continue
                start_time = time.time()
                img = letter_box(im=frame.copy(), new_shape=(MODEL_SIZE[1], MODEL_SIZE[0]), pad_color=(0, 0, 0))
                input = np.expand_dims(img, axis=0)
                outputs = self.rknn_lite.inference([input])
                boxes, classes, scores = post_process(outputs)
                img_p = frame.copy()
                detections = {}
                frame_info = []
                if boxes is not None:
                    for box, cl, score in zip(boxes, classes, scores):
                        class_name = self.class_names[cl]
                        detections[class_name] = detections.get(class_name, 0) + 1
                        all_detections[class_name] = all_detections.get(class_name, 0) + 1
                        frame_info.append(f"class={class_name}, id={cl}, score={score:.4f}")
                    draw(img_p, boxes, scores, classes, self.class_names, self.color_palette)
                if out:
                    out.write(img_p)
                frame_time = (time.time() - start_time) * 1000
                frame_times.append(frame_time)
                frame_idx += 1
                print(f"Frame {frame_idx}: " + (", ".join(frame_info) if frame_info else "No objects detected"))
        finally:
            self.stop_flag = True
            reader_thread.join()
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        total_time = sum(frame_times)
        average_time = total_time / len(frame_times) if frame_times else 0
        if show_progress:
            print(f"\nRTSP流处理完成。总帧数: {frame_idx}")
            if frame_times:
                print(f"平均每帧处理时间: {average_time:.2f} ms")
                print("检测统计:")
                for cls, count in all_detections.items():
                    print(f"  {cls}: {count}")
        return {
            'total_frames': frame_idx,
            'total_time_seconds': total_time / 1000.0,
            'average_time_ms': average_time,
            'detections': all_detections
        }

    def process_rtsp_iter(self, rtsp_url, show_progress=True):
        """
        RTSP流推理生成器，每帧yield详细检测结果
        :param rtsp_url: RTSP流地址
        :param show_progress: 是否显示进度
        :yield: 每帧的检测结果字典
        """
        print(f"尝试打开RTSP流: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"无法打开RTSP流: {rtsp_url}")
            return
        print("RTSP流打开成功")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = None
        frame_idx = 0
        self.stop_flag = False
        self.queue.queue.clear()
        reader_thread = threading.Thread(target=self._frame_reader, args=(cap,))
        reader_thread.start()
        try:
            while True:
                try:
                    frame = self.queue.get(timeout=1)
                except queue.Empty:
                    if not reader_thread.is_alive():
                        break
                    continue
                start_time = time.time()
                img = letter_box(im=frame.copy(), new_shape=(MODEL_SIZE[1], MODEL_SIZE[0]), pad_color=(0, 0, 0))
                input = np.expand_dims(img, axis=0)
                outputs = self.rknn_lite.inference([input])
                boxes, classes, scores = post_process(outputs)
                img_p = frame.copy()
                detections = {}
                frame_info = []
                if boxes is not None:
                    for box, cl, score in zip(boxes, classes, scores):
                        class_name = self.class_names[cl]
                        detections[class_name] = detections.get(class_name, 0) + 1
                        frame_info.append({
                            'class': class_name,
                            'id': int(cl),
                            'score': float(score),
                            'bbox': [float(x) for x in box]
                        })
                    draw(img_p, boxes, scores, classes, self.class_names, self.color_palette)
                frame_time = (time.time() - start_time) * 1000
                frame_idx += 1
                if show_progress:
                    print(f"Frame {frame_idx}: " + (", ".join([f"class={fi['class']}, id={fi['id']}, score={fi['score']:.4f}" for fi in frame_info]) if frame_info else "No objects detected"))
                yield {
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'img_p': img_p,
                    'boxes': boxes,
                    'classes': classes,
                    'scores': scores,
                    'detections': detections,
                    'frame_info': frame_info,
                    'frame_time': frame_time
                }
        finally:
            self.stop_flag = True
            reader_thread.join()
            cap.release()
            cv2.destroyAllWindows()

    def release(self):
        """释放RKNN资源"""
        self.rknn_lite.release()

# ================== FrameProcessor 封装 ==================
class FrameProcessor:
    """
    帧处理器类，兼容YOLO风格接口，封装RKNN推理
    """
    def __init__(self, model_path, classes_file=None):
        """
        初始化帧处理器
        :param model_path: RKNN模型路径
        :param classes_file: 类别文件路径(可选)
        """
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            self.class_names = DEFAULT_CLASSES
        self.coco_id_list = DEFAULT_COCO_ID_LIST
        self.model_path = model_path
        self.rknn_lite = RKNNLite()
        self.color_palette = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self._init_model()

    def _init_model(self):
        """加载并初始化RKNN模型"""
        ret = self.rknn_lite.load_rknn(self.model_path)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)

    def process_frame(self, frame):
        """
        处理单帧图像，返回检测结果
        :param frame: 输入图像帧
        :return: 处理结果字典
        """
        import time
        start_time = time.time()
        img = letter_box(im=frame.copy(), new_shape=(MODEL_SIZE[1], MODEL_SIZE[0]), pad_color=(0, 0, 0))
        input = np.expand_dims(img, axis=0)
        outputs = self.rknn_lite.inference([input])
        boxes, classes, scores = post_process(outputs)
        img_p = frame.copy()
        detections = {}
        box_list = []
        if boxes is not None:
            for box, cl, score in zip(boxes, classes, scores):
                class_name = self.class_names[cl]
                detections[class_name] = detections.get(class_name, 0) + 1
                box_list.append({
                    'class': class_name,
                    'confidence': float(score),
                    'bbox': [float(x) for x in box]
                })
            draw(img_p, boxes, scores, classes, self.class_names, self.color_palette)
        processing_time = (time.time() - start_time) * 1000
        return {
            'frame': img_p,
            'detections': detections,
            'boxes': box_list,
            'processing_time_ms': processing_time
        }

# ------------------- 主入口 -------------------
def main():
    """
    命令行主入口，处理参数并启动RTSP推理
    """
    parser = argparse.ArgumentParser(description='RKNN YOLOv8 RTSP流推理工具')
    parser.add_argument('--model', type=str, default=DEFAULT_RKNN_MODEL, help='RKNN模型路径')
    parser.add_argument('--input', type=str, required=True, help='RTSP流地址（rtsp://...）')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径（可选）')
    parser.add_argument('--classes', type=str, default=None, help='类别文件（每行一个类别）')
    parser.add_argument('--coco_ids', type=str, default=None, help='coco id列表文件（每行一个id）')
    parser.add_argument('--show_progress', action='store_true', help='显示进度')
    args = parser.parse_args()
    if not args.input.lower().startswith('rtsp://'):
        print('只支持rtsp流输入！请使用 --input rtsp://...')
        return
    if args.classes and os.path.exists(args.classes):
        with open(args.classes, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = DEFAULT_CLASSES
    if args.coco_ids and os.path.exists(args.coco_ids):
        with open(args.coco_ids, 'r') as f:
            coco_id_list = [int(line.strip()) for line in f.readlines()]
    else:
        coco_id_list = DEFAULT_COCO_ID_LIST
    processor = RKNNRTSPProcessor(args.model, class_names, coco_id_list, save_path=args.output)
    result = processor.process_rtsp(args.input, args.show_progress)
    print("\n推理统计结果:")
    print(result)
    processor.release()

if __name__ == '__main__':
    main()