import cv2

"""
摄像头/RTSP视频流采集与显示脚本

功能：
- 连接RTSP摄像头流
- 实时显示视频画面
- 支持ESC键退出

依赖：
- OpenCV (cv2)：视频流采集与显示
"""

# 摄像头 RTSP 地址（高清码流 /11）
rtsp_url = "rtsp://192.168.0.101/11"

# 打开视频流
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("错误：无法连接到摄像头")
    exit()

print("成功连接到摄像头，按 ESC 键退出窗口")

while True:
    ret, frame = cap.read()
    if not ret:
        print("警告：无法获取画面")
        break

    # 实时显示画面（可在此处插入YOLO检测等处理）
    cv2.imshow("Tomato Camera Stream", frame)

    # 按 ESC 键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()