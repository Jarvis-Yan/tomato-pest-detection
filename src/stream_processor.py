#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频流处理与服务器通信系统

此系统专注于视频流的处理和服务器通信，主要特性：
- 视频流管理（文件/CSI摄像头）
- 服务器通信
- 数据上传管理
- 实时监控

技术特点：
- 支持多种视频源
- 低延迟数据传输
- JSON格式通信
- 多线程处理

处理流程：
1. 初始化：配置服务器连接、创建处理器
2. 流处理：读取视频流、处理帧
3. 数据管理：队列管理、上传调度
4. 通信：与服务器交互

依赖：
- OpenCV (cv2): 视频处理
- requests: HTTP通信
- process_video: 帧处理模块
"""

import cv2
import json
import time
import requests
import threading
import queue
from yolov8 import FrameProcessor
import os


class StreamProcessor:
    def __init__(self, server_url, model_path, classes_file):
        """
        初始化流处理器

        参数:
        - server_url: 服务器基础URL
        - model_path: YOLO模型路径
        - classes_file: 类别文件路径
        """
        self.server_url = server_url
        self.local_url = "http://localhost:8000"  # 本地服务器URL
        self.frame_processor = FrameProcessor(model_path, classes_file)

        # 两个队列：帧队列和检测信息队列
        self.frame_queue = queue.Queue(maxsize=2)  # 极致实时：只保留最新帧
        self.detection_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=10)  # 新增显示队列

        # 状态控制
        self.running = False
        self.upload_threads = []
        self.display_thread = None  # 显示线程

        # 检测统计
        self.detection_stats = {}

    def start(self, input_source):
        """
        启动流处理

        参数:
        - input_source: 视频源（文件路径或CSI摄像头索引）
        """
        if self.running:
            print("处理器已经在运行")
            return

        self.running = True

        # 上传线程数增至8
        self.upload_threads = [
            threading.Thread(target=self._upload_frame_worker, args=(self.server_url, "远程"))
            for _ in range(8)
        ] + [
            threading.Thread(target=self._upload_detection_worker, args=(self.server_url, "远程"))
            for _ in range(4)
        ]
        for thread in self.upload_threads:
            thread.start()

        # 启动显示线程
        self.display_thread = threading.Thread(target=self._display_worker)
        self.display_thread.start()

        # 启动流处理线程
        stream_thread = threading.Thread(target=self._process_stream, args=(input_source,))
        stream_thread.start()

        return stream_thread

    def stop(self):
        """停止流处理"""
        self.running = False
        for thread in self.upload_threads:
            thread.join()
        if self.display_thread:
            self.display_thread.join()

    def _process_stream(self, input_source):
        """
        处理视频流

        参数:
        - input_source: 视频源（文件路径或CSI摄像头索引）
        """
        disease_types = {
            'mosaic_virus', 'early_blight', 'late_blight', 'septoria',
            'yellow_leaf_curl_virus', 'leaf_mold', 'leaf_miner', 'spider_mites'
        }
        # 优化RTSP流读取
        if isinstance(input_source, str) and input_source.lower().startswith('rtsp://'):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(input_source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        else:
            cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {input_source}")
            return
        try:
            frame_count = 0
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                result = self.frame_processor.process_frame(frame)
                # 推理后帧放入显示队列
                if not self.display_queue.full():
                    self.display_queue.put(result['frame'])
                detected_diseases = {
                    cls: count for cls, count in result['detections'].items()
                    if cls in disease_types and count > 0
                }
                # 极致实时：只保留最新帧
                while self.frame_queue.qsize() > 0:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                self.frame_queue.put(result['frame'])
                # 优化：检测信息入队前丢弃最旧，只保留最新
                if detected_diseases:
                    while self.detection_queue.full():
                        try:
                            self.detection_queue.get_nowait()
                        except queue.Empty:
                            break
                    detection_data = {
                        'timestamp': time.time(),
                        'frame_number': frame_count,
                        'detections': result['detections'],
                        'boxes': result['boxes'],
                        'stats': self.detection_stats,
                        'disease_info': [
                            {
                                'name': cls,
                                'count': count,
                                'confidence': max([box['confidence'] for box in result['boxes'] if box['class'] == cls], default=0.0)
                            }
                            for cls, count in detected_diseases.items()
                        ]
                    }
                    self.detection_queue.put(detection_data)
                print(f"处理进度: {frame_count} 帧 | 当前帧处理时间: {result['processing_time_ms']:.2f} ms | 检测状态: {'发现疾病' if detected_diseases else '无疾病'} | 疾病统计: {[f'{cls}:{count}' for cls, count in detected_diseases.items()]}", end='\r', flush=True)
        finally:
            cap.release()

    def _upload_frame_worker(self, server_url, server_type):
        """
        上传帧到服务器

        参数:
        - server_url: 服务器URL
        - server_type: 服务器类型（用于日志显示）
        """
        while self.running or not self.frame_queue.empty():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    self._upload_frame(frame, server_url, server_type)
                time.sleep(0.01)  # 避免CPU占用过高
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"{server_type}服务器帧上传工作线程错误: {str(e)}")
                time.sleep(1)

    def _upload_detection_worker(self, server_url, server_type):
        """
        上传检测结果到服务器

        参数:
        - server_url: 服务器URL
        - server_type: 服务器类型（用于日志显示）
        """
        while self.running or not self.detection_queue.empty():
            try:
                if not self.detection_queue.empty():
                    data = self.detection_queue.get_nowait()
                    self._upload_detection(data, server_url, server_type)
                time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"{server_type}服务器检测结果上传工作线程错误: {str(e)}")
                time.sleep(1)

    def _upload_frame(self, frame, server_url, server_type):
        """
        上传帧到服务器

        参数:
        - frame: 要上传的帧
        - server_url: 服务器URL
        - server_type: 服务器类型（用于日志显示）
        """
        try:
            if not hasattr(self, '_upload_frame_saved'):
                cv2.imwrite("debug_upload.jpg", frame)
                self._upload_frame_saved = True
            frame = cv2.resize(frame, (640, 480))  # 建议注释掉
            # 极致压缩：图片压缩率调为50
            _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            response = requests.post(
                f"{server_url}/upload",
                data=img_encoded.tobytes(),
                headers={'Content-Type': 'image/jpeg'},
                timeout=2
            )
            if response.status_code != 200:
                print(f"\n{server_type}服务器帧上传失败: {response.status_code}")
            else:
                print(f"\n{server_type}服务器帧上传成功")
        except Exception as e:
            print(f"\n{server_type}服务器帧上传错误: {str(e)}")

    def _upload_detection(self, data, server_url, server_type):
        """
        上传检测结果到服务器

        参数:
        - data: 包含检测信息的字典
        - server_url: 服务器URL
        - server_type: 服务器类型（用于日志显示）
        """
        try:
            # 将数据转换为JSON字符串
            json_data = json.dumps(data)

            print(f"\n尝试上传检测信息到{server_type}服务器: {server_url}/detection")
            print(f"检测到的疾病: {data['disease_info']}")

            response = requests.post(
                f"{server_url}/detection",
                data=json_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )

            if response.status_code != 200:
                print(f"\n{server_type}服务器检测结果上传失败: {response.status_code}")
                print(f"请求URL: {server_url}/detection")
            else:
                print(f"\n{server_type}服务器检测结果上传成功")
        except Exception as e:
            print(f"\n{server_type}服务器检测结果上传错误: {str(e)}")
            print(f"请求URL: {server_url}/detection")

    def _display_worker(self):
        # 设置窗口为全屏
        cv2.namedWindow("推理结果实时画面", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("推理结果实时画面", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        screen_width, screen_height = 1280, 720  # 如需自适应可进一步调整
        while self.running or not self.display_queue.empty():
            try:
                if not self.display_queue.empty():
                    frame = self.display_queue.get_nowait()
                    # resize到全屏分辨率
                    frame_resized = cv2.resize(frame, (screen_width, screen_height))
                    cv2.imshow("推理结果实时画面", frame_resized)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"显示线程错误: {str(e)}")
                time.sleep(1)
        cv2.destroyAllWindows()


def main():
    # 配置参数 - 确保与video_processor.py完全相同
    server_url = "http://120.27.153.89:8000"
    model_path = "/home/elf/tfcard/yolov8_test/yolov8.rknn"
    classes_file = "/home/elf/tfcard/yolov8_test/classes.txt"

    print(f"使用服务器URL: {server_url}")

    # 创建处理器实例
    processor = StreamProcessor(server_url, model_path, classes_file)

    try:
        # 启动处理
        stream_thread = processor.start("rtsp://192.168.0.101:554/11")
        stream_thread.join()
    except KeyboardInterrupt:
        print("\n正在停止处理...")
    finally:
        processor.stop()


if __name__ == "__main__":
    main() 