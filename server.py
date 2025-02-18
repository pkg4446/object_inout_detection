# 간단한 HTTP 웹서버를 파이썬으로 구현하는 방법

import http.server
import socketserver
import cv2
import json

# JSON 파일에서 ROI1_RATIO 값을 불러오기
with open('ROI.json', 'r') as file:
    data = json.load(file)
    ROIs = [tuple(data.get("ROI1_RATIO", (0.0, 0.0, 0.0, 0.0))), 
            tuple(data.get("ROI2_RATIO", (0.0, 0.0, 0.0, 0.0)))]
                                               
# 포트 번호 설정
PORT = 3000

class VideoStreamHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # ROI로 박스 그리기
                h, w, _ = frame.shape
                for ROI, color in zip(ROIs, [(255, 0, 0), (0, 255, 0)]):
                    start_point = (int(ROI[0] * w), int(ROI[1] * h))
                    end_point = (int((ROI[0] + ROI[2]) * w), int((ROI[1] + ROI[3]) * h))
                    thickness = 2 # 두께
                    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

                _, buffer = cv2.imencode('.jpg', frame)
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            super().do_GET()

Handler = VideoStreamHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("서버가 포트 http://localhost:{} 에서 시작되었습니다.".format(PORT))
    # 서버 시작
    httpd.serve_forever()