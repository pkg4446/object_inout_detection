import cv2
import os
import datetime
import time

# 제외할 카메라 인덱스를 저장할 파일
EXCLUDED_CAMERAS_FILE = "camera_out.txt"

def get_connected_cameras():
    i = 0
    num_cameras = 0
    while True:
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        num_cameras += 1
        cap.release()
        i += 1
    return num_cameras

def get_excluded_cameras():
    excluded_indices = set()
    if os.path.exists(EXCLUDED_CAMERAS_FILE):
        with open(EXCLUDED_CAMERAS_FILE, 'r') as f:
            for line in f:
                try:
                    index = int(line.strip())
                    excluded_indices.add(index)
                except ValueError:
                    continue # 숫자가 아닌 줄은 무시
    return excluded_indices

def capture_snapshots_with_preview(interval_ms=50):
    num_cameras = get_connected_cameras()
    if num_cameras == 0:
        print("연결된 카메라를 찾을 수 없습니다.")
        return

    excluded_indices = get_excluded_cameras()
    print(f"제외할 카메라 인덱스: {excluded_indices}")

    print(f"{num_cameras}개의 카메라가 연결되어 있습니다.")

    caps = []
    active_camera_indices = []

    # 프리뷰 창의 원하는 너비와 높이 설정
    preview_width = 480
    preview_height = 320

    for i in range(num_cameras):
        if i in excluded_indices:
            print(f"카메라 {i}는 제외 목록에 있으므로 건너뜁니다.")
            caps.append(None) # None을 추가하여 인덱스 유지
            continue

        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"카메라 {i}를 열 수 없습니다. 다음 카메라로 넘어갑니다.")
            caps.append(None)
            continue
        
        # FHD (1920x1080) 해상도 설정
        # 카메라가 해당 해상도를 지원하지 않으면, 지원하는 가장 가까운 해상도로 설정됩니다.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        caps.append(cap)
        active_camera_indices.append(i)

    if not active_camera_indices:
        print("활성화된 카메라가 없습니다.")
        return

    # 기본 저장 디렉토리 (현재 날짜 기준)
    now = datetime.datetime.now()
    base_output_dir = now.strftime(".\\%Y\\%m\\%d")
    print(f"스냅샷은 '{base_output_dir}\\camera_index' 폴더에 저장됩니다.")

    # 각 카메라별 폴더 생성
    for idx in active_camera_indices:
        camera_output_dir = os.path.join(base_output_dir, f"camera_{idx}")
        os.makedirs(camera_output_dir, exist_ok=True)

    print("스냅샷 캡처 및 프리뷰를 시작합니다. 종료하려면 'q'를 누르세요.")
    
    last_capture_time = time.time() * 1000 # 밀리초 단위로 저장

    try:
        while True:
            current_time = time.time() * 1000
            
            # 모든 활성화된 카메라에서 프레임 읽기 및 화면 표시
            for i, cap in enumerate(caps):
                if cap is not None:
                    ret, frame = cap.read()
                    if ret:
                        # --- 여기부터 프리뷰 크기 조절 추가 ---
                        resized_frame = cv2.resize(frame, (preview_width, preview_height))
                        cv2.imshow(f'Camera {i} Preview', resized_frame)
                        # --- 프리뷰 크기 조절 끝 ---
                    else:
                        print(f"카메라 {i}에서 프레임을 캡처할 수 없습니다.")
                        # 카메라가 연결 해제되었을 경우 처리
                        if cap.isOpened(): # 아직 열려있다면 계속 시도, 아니라면 종료
                            pass 
                        else:
                            print(f"카메라 {i} 연결이 끊어졌습니다. 프리뷰를 닫습니다.")
                            cv2.destroyWindow(f'Camera {i} Preview')
                            caps[i] = None # 해당 카메라 비활성화
                            if i in active_camera_indices:
                                active_camera_indices.remove(i)

            # 스냅샷 저장 조건 확인 (0.05초 간격)
            if current_time - last_capture_time >= interval_ms:
                for i, cap in enumerate(caps):
                    if cap is not None: # 활성화된 카메라만 저장
                        ret, frame = cap.read() # 스냅샷을 위해 다시 프레임 읽기 (최신 프레임)
                        if ret:
                            timestamp = datetime.datetime.now().strftime("%H_%M_%S_%f")[:-3] # 밀리초까지
                            camera_output_dir = os.path.join(base_output_dir, f"camera_{i}")
                            filename = os.path.join(camera_output_dir, f"{timestamp}.jpg") 
                            cv2.imwrite(filename, frame)
                last_capture_time = current_time

            # 키보드 입력 대기 (1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        print("스냅샷 캡처 및 프리뷰가 종료되었습니다.")

if __name__ == "__main__":
    capture_snapshots_with_preview(interval_ms=50)