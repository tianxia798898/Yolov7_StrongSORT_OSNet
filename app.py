from flask import Flask, Response, request, jsonify
from flask_socketio import SocketIO
import cv2
from yolov7 import YOLOv7
from strong_sort import StrongSORT

app = Flask(__name__)
socketio = SocketIO(app)

# 初始化模型
detector = YOLOv7(model_path="yolov7.pt", device="cuda")
tracker = StrongSORT(model_path="osnet_x1_0_market1501.pt", device="cuda")

# 视频处理函数
def generate_frames(video_source):
    cap = cv2.VideoCapture(video_source)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 检测和跟踪
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame)
        
        # 绘制结果
        for obj in tracked_objects:
            x1, y1, x2, y2, id = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 转为视频流格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 视频流接口
@app.route('/video_feed')
def video_feed():
    video_source = request.args.get("source", 0)  # 默认摄像头
    return Response(generate_frames(video_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 参数设置接口
@app.route('/set_params', methods=['POST'])
def set_params():
    params = request.json
    # 设置算法参数，如检测阈值、视频源等
    detector.set_threshold(params.get("threshold", 0.5))
    return jsonify({"status": "success", "params": params})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)