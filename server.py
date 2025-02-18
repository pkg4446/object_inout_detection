import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import http.server
import socketserver
import cv2
import time
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor

# JSON 파일에서 ROI1_RATIO 값을 불러오기
with open('ROI.json', 'r') as file:
    data = json.load(file)
    ROIs = [tuple(data.get("ROI1_RATIO", (0.0, 0.0, 0.0, 0.0))), 
            tuple(data.get("ROI2_RATIO", (0.0, 0.0, 0.0, 0.0)))]

# 포트 번호 설정
PORT = 3000

@dataclass
class Config:
    MODEL_PATH = "./model/detect.tflite"
    LABEL_PATH = "./model/labelmap.txt"
    CONFIDENCE_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.5
    FRAME_RESET_COUNT = 30
    NUM_THREADS = 4

class BeeDetector:
    def __init__(self, config: Config):
        self.config = config
        self.interpreter = self._load_model()
        self.labels = self._load_labels()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.executor = ThreadPoolExecutor(max_workers=self.config.NUM_THREADS)

    def _load_model(self) -> tf.lite.Interpreter:
        interpreter = tf.lite.Interpreter(self.config.MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter

    def _load_labels(self) -> List[str]:
        with open(self.config.LABEL_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        input_size = (self.input_details[0]['shape'][2], 
                     self.input_details[0]['shape'][1])
        image = Image.fromarray(frame)
        image = image.resize(input_size)
        return np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

    def perform_detection(self, frame: np.ndarray) -> List[Tuple[List[int], float, str]]:
        input_data = self.preprocess_image(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        detections = []
        for idx, score in enumerate(scores):
            if score >= self.config.CONFIDENCE_THRESHOLD:
                ymin, xmin, ymax, xmax = boxes[idx]
                left = int(xmin * frame.shape[1])
                right = int(xmax * frame.shape[1])
                top = int(ymin * frame.shape[0])
                bottom = int(ymax * frame.shape[0])
                detections.append(([left, top, right - left, bottom - top], 
                                 score, 
                                 self.labels[int(classes[idx])]))
        if detections:
            boxes_list = [d[0] for d in detections]
            scores_list = [d[1] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, 
                                     self.config.CONFIDENCE_THRESHOLD, 
                                     self.config.IOU_THRESHOLD)
            return [detections[i] for i in indices.flatten()]
        return []
    
class VideoStreamHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.config = Config()
        self.detector = BeeDetector(self.config)
        super().__init__(*args, **kwargs)
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            cap = cv2.VideoCapture(0)

            frame_count = 0
            start_time  = time.time()
            frame_per_sec = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # 1초마다 FPS 계산
                    frame_per_sec = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()

                # ROI로 박스 그리기
                h, w, _ = frame.shape
                for ROI, color in zip(ROIs, [(255, 0, 0), (0, 255, 0)]):
                    start_point = (int(ROI[0] * w), int(ROI[1] * h))
                    end_point = (int((ROI[0] + ROI[2]) * w), int((ROI[1] + ROI[3]) * h))
                    thickness = 2
                    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
                # 객체 감지 수행
                detections = self.detector.perform_detection(frame)
                for box, score, label in detections:
                    x, y, width, height = box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {score:.2f}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # 카운트 정보 표시
                cv2.putText(frame, f"FPS: {frame_per_sec:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # 프레임을 JPEG로 인코딩하여 전송
                _, buffer = cv2.imencode('.jpg', frame)
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            super().do_GET()
# 서버 시작
Handler = VideoStreamHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"서버가 포트 http://localhost:{PORT} 에서 시작되었습니다.")
    httpd.serve_forever()