import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import tensorflow as tf
import numpy as np
import cv2
import time
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import threading

@dataclass
class Config:
    def __init__(self, config_path: str = None):
        # 설정값을 관리하는 클래스
        self.MODEL_PATH = "./model/detect.tflite"
        self.LABEL_PATH = "./model/labelmap.txt"
        self.VIDEO_PATH = "./video/test.avi"
        self.CONFIDENCE_THRESHOLD = 0.2
        self.IOU_THRESHOLD = 0.5
        self.ROI1_RATIO = (0.0, 0.0, 0.0, 0.0)
        self.ROI2_RATIO = (0.0, 0.0, 0.0, 0.0)
        self.FRAME_RESET_COUNT = 30
        self.NUM_THREADS = 4 # 병렬 처리 스레드 수

        # JSON 파일에서 값 불러오기
        if config_path:
            self.load_from_json(config_path)

    def load_from_json(self, config_path: str):
        with open(config_path, 'r') as file:
            data = json.load(file)
            self.ROI1_RATIO = tuple(data.get("ROI1_RATIO", (0.0, 0.0, 0.0, 0.0)))
            self.ROI2_RATIO = tuple(data.get("ROI2_RATIO", (0.0, 0.0, 0.0, 0.0)))

class BeeDetector:
    def __init__(self, config: Config):
        self.config = config
        self.interpreter = self._load_model()
        self.labels = self._load_labels()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 상태 변수 초기화
        self.bee_counts = {"in": 0, "out": 0, "sum": 0}
        self.LOI_counts = {
            "bee_in": 0, "bee_out": 0,
            "in": [0, 0], "chk": [0, 0], "out": [0, 0]
        }
        self.frame_count = 0
        self.executor = ThreadPoolExecutor(max_workers=self.config.NUM_THREADS)
        self.server_url = "http://localhost:3002/"  # 웹 서버 URL
        self._start_post_timer()

    def _start_post_timer(self):
        # 5분마다 bee_counts를 전송하고 초기화하는 타이머 시작
        self.post_timer = threading.Timer(60*5, self._post_bee_counts)
        self.post_timer.start()

    def _post_bee_counts(self):
        # bee_counts를 웹 서버로 전송하고 초기화
        try:
            response = requests.post(self.server_url, json=self.bee_counts)
            response.raise_for_status()
            print("bee_counts 전송 성공:", self.bee_counts)
        except requests.RequestException as e:
            print("bee_counts 전송 실패:", e)
        # bee_counts 초기화
        self.bee_counts = {"in": 0, "out": 0, "sum": 0}
        # 타이머 재시작
        self._start_post_timer()

    def _load_model(self) -> tf.lite.Interpreter:
        # TFLite 모델 로드
        interpreter = tf.lite.Interpreter(self.config.MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter

    def _load_labels(self) -> List[str]:
        # 라벨 파일 로드
        with open(self.config.LABEL_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        # 이미지 전처리
        input_size = (self.input_details[0]['shape'][2], 
                     self.input_details[0]['shape'][1])
        image = Image.fromarray(frame)
        image = image.resize(input_size)
        return np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

    def _update_bee_tracking(self, bee_positions: List[Tuple[int, int]], 
                           roi1: Tuple[int, int, int, int], 
                           roi2: Tuple[int, int, int, int]):
        # 벌 추적 상태 업데이트
        if bee_positions:
            # 이전 상태 업데이트
            self.LOI_counts["in"] = [self.LOI_counts["in"][1], 0]
            self.LOI_counts["chk"] = [self.LOI_counts["chk"][1], 0]
            self.LOI_counts["out"] = [self.LOI_counts["out"][1], 0]
            # 현재 위치 기반 카운트
            for x, y in bee_positions:
                in_roi1 = (roi1[0] <= x <= roi1[0] + roi1[2] and 
                          roi1[1] <= y <= roi1[1] + roi1[3])
                in_roi2 = (roi2[0] <= x <= roi2[0] + roi2[2] and 
                          roi2[1] <= y <= roi2[1] + roi2[3])
                if in_roi1:
                    self.LOI_counts["in"][1] += 1
                elif in_roi2:
                    self.LOI_counts["chk"][1] += 1
                else:
                    self.LOI_counts["out"][1] += 1
            # 상태 변화 감지 및 처리
            if (self.LOI_counts["in"][0] != self.LOI_counts["in"][1] or 
                self.LOI_counts["chk"][0] != self.LOI_counts["chk"][1] or 
                self.LOI_counts["out"][0] != self.LOI_counts["out"][1]):
                
                self._process_state_change()
            
            self.frame_count = 0
        
        elif self.frame_count > self.config.FRAME_RESET_COUNT:
            # 프레임 리셋
            self.frame_count = 0
            self.LOI_counts["bee_in"]  = 0
            self.LOI_counts["bee_out"] = 0
            self.LOI_counts["in"] = [self.LOI_counts["in"][1], self.LOI_counts["in"][1]]
            self.LOI_counts["chk"] = [self.LOI_counts["chk"][1], self.LOI_counts["chk"][1]]
            self.LOI_counts["out"] = [self.LOI_counts["out"][1], self.LOI_counts["out"][1]]
            print("감지된 벌 없음")
        else:
            self.frame_count += 1

    def _process_state_change(self):
        # 상태 변화 처리
        # ROI2(중간 영역) 벌 수 감소
        if self.LOI_counts["chk"][1] < self.LOI_counts["chk"][0]:
            self.frame_count = 0
            if self.LOI_counts["out"][1] > self.LOI_counts["out"][0]:
                if self.LOI_counts["bee_out"] > 0:
                    self.LOI_counts["bee_out"] -= 1
                    self.update_bee_counts(False)  # 퇴장
                else:
                    self.LOI_counts["bee_in"] -= 1
                    print("2차 퇴장")
            elif self.LOI_counts["in"][1] > self.LOI_counts["in"][0]:
                if self.LOI_counts["bee_in"] > 0:
                    self.LOI_counts["bee_in"] -= 1
                    self.update_bee_counts(True)  # 입장
                else:
                    self.LOI_counts["bee_out"] -= 1
                    print("2차 입장")
        # ROI2(중간 영역) 벌 수 증가
        elif self.LOI_counts["chk"][1] > self.LOI_counts["chk"][0]:
            self.frame_count = 0
            if self.LOI_counts["out"][1] < self.LOI_counts["out"][0]:
                self.LOI_counts["bee_in"] += 1
                print("입장 준비")
            elif self.LOI_counts["in"][1] < self.LOI_counts["in"][0]:
                self.LOI_counts["bee_out"] += 1
                print("퇴장 준비")
        # ROI2(중간 영역) 벌 수 유지
        else:
            print("event")
            if (self.LOI_counts["bee_out"] > 0 and 
                self.LOI_counts["out"][1] > self.LOI_counts["out"][0]):
                self.update_bee_counts(False)
            if (self.LOI_counts["bee_in"] > 0 and 
                self.LOI_counts["in"][1] > self.LOI_counts["in"][0]):
                self.update_bee_counts(True)

    def perform_detection(self, frame: np.ndarray) -> List[Tuple[List[int], float, str]]:
        # 객체 감지 수행
        input_data = self.preprocess_image(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        # 결과 가져오기
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        # NMS 적용을 위한 박스 변환
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
        # NMS 적용
        if detections:
            boxes_list = [d[0] for d in detections]
            scores_list = [d[1] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, 
                                     self.config.CONFIDENCE_THRESHOLD, 
                                     self.config.IOU_THRESHOLD)
            return [detections[i] for i in indices.flatten()]
        return []

    def update_bee_counts(self, in_count: bool):
        # 벌 카운트 업데이트
        if in_count:
            self.bee_counts["sum"] += 1
            self.bee_counts["in"] += 1
            print("벌 입장!")
        else:
            self.bee_counts["sum"] -= 1
            self.bee_counts["out"] += 1
            print("벌 퇴장!")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # 프레임 처리
        height, width = frame.shape[:2]
        # ROI 영역 계산 및 그리기
        roi1 = self._calculate_roi(width, height, self.config.ROI1_RATIO)
        roi2 = self._calculate_roi(width, height, self.config.ROI2_RATIO)

        frame = self._draw_rois(frame, roi1, roi2)
        # 병렬 처리로 객체 감지 수행
        future = self.executor.submit(self.perform_detection, frame)
        detections = future.result()
        # 감지된 객체 표시 및 위치 추적
        bee_positions = []
        for box, score, label in detections:
            frame = self._draw_detection(frame, box, score, label)
            bee_positions.append((box[0] + box[2]//2, box[1] + box[3]//2))
        self._update_bee_tracking(bee_positions, roi1, roi2)
        
        return frame

    def _calculate_roi(self, width: int, height: int, ratio: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        # ROI 영역 계산
        return (int(ratio[0] * width), int(ratio[1] * height),
                int(ratio[2] * width), int(ratio[3] * height))

    def _draw_rois(self, frame: np.ndarray, roi1: Tuple[int, int, int, int], 
                   roi2: Tuple[int, int, int, int]) -> np.ndarray:
        return frame

    def _draw_detection(self, frame: np.ndarray, box: List[int], 
                       score: float, label: str) -> np.ndarray:
        return frame

    def run(self):
        # 비디오 처리 실행
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        if not cap.isOpened():
            raise ValueError("비디오를 열 수 없습니다.")

        try:
            frame_count = 0
            start_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.process_frame(frame)

                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # 1초마다 FPS 계산
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=True)
            self.post_timer.cancel()  # 타이머 취소

def main():
    config = Config("ROI.json")
    detector = BeeDetector(config)
    detector.run()

if __name__ == "__main__":
    main()