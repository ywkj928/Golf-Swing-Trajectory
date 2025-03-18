import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import math

# YOLO 모델 로드
model = YOLO("source/best-segv1.pt")  # 모델 경로를 넣어주세요

# 비디오 파일 경로 (로컬 동영상 경로를 지정하세요)
video_path = "source/data/dtl-02.01.2025-152209.mp4"  # 여기에 비디오 파일 경로를 넣어주세요

# 비디오 캡처
cap = cv2.VideoCapture(video_path)

# 비디오가 제대로 열렸는지 확인
if not cap.isOpened():
    print("Error: 비디오 파일을 열 수 없습니다.")
    exit()

# 총 프레임 수를 가져오기 전에 비디오 파일이 제대로 열렸는지 다시 한번 확인
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if frame_count == 0:
    print("Error: 비디오에서 프레임을 읽을 수 없습니다.")
    exit()

# 추적 경로를 저장할 딕셔너리 (track_id -> [(x, y), ...])
track_history = defaultdict(list)

# 트랙바를 위한 윈도우 이름 설정
cv2.namedWindow("Video Playback")

# 트랙바를 생성 (슬라이더의 범위는 0 ~ 100으로 설정, 전체 길이에 비례한 비율을 설정)
cv2.createTrackbar("Progress", "Video Playback", 0, frame_count - 1, lambda x: None)

# 밝기 조정용 트랙바 (0 ~ 200)
cv2.createTrackbar("Brightness", "Video Playback", 100, 200, lambda x: None)

# 명암 대비 조정용 트랙바 (0 ~ 100)
cv2.createTrackbar("Contrast", "Video Playback", 50, 100, lambda x: None)

# 신뢰도 조정용 트랙바 (0.0 ~ 1.0)
cv2.createTrackbar("Confidence Threshold", "Video Playback", 50, 100, lambda x: None)

# 추적할 클래스 이름 설정 (head, hand)
class_names = ['head', 'hand']

# 속도 예측을 위한 함수 (단순한 선형 예측 사용)
def predict_position(last_position, velocity, delta_t=1):
    """
    현재 위치와 속도를 기반으로 물체의 미래 위치를 예측합니다.
    :param last_position: 마지막 위치 (x, y)
    :param velocity: 속도 벡터 (vx, vy)
    :param delta_t: 예측할 시간 간격 (기본값 1 프레임)
    :return: 예측된 위치 (x', y')
    """
    x, y = last_position
    vx, vy = velocity
    # 새로운 위치 예측
    new_x = x + vx * delta_t
    new_y = y + vy * delta_t
    return new_x, new_y

# 두 점 사이의 유클리드 거리 계산
def calculate_distance(point1, point2):
    """
    두 점 사이의 유클리드 거리를 계산합니다.
    :param point1: 첫 번째 점 (x1, y1)
    :param point2: 두 번째 점 (x2, y2)
    :return: 두 점 사이의 거리
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while cap.isOpened():
    # 트랙바 값 읽기
    brightness = cv2.getTrackbarPos("Brightness", "Video Playback") - 100  # 0 ~ 100에서 -100으로 조정
    contrast = cv2.getTrackbarPos("Contrast", "Video Playback") / 50.0  # 0 ~ 100에서 0 ~ 2.0으로 변환
    confidence_threshold = cv2.getTrackbarPos("Confidence Threshold", "Video Playback") / 100.0  # 0 ~ 100에서 0.0 ~ 1.0으로 변환
    progress = cv2.getTrackbarPos("Progress", "Video Playback")  # 슬라이더로 프레임 이동

    # 비디오 파일에서 해당 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, progress)

    success, frame = cap.read()
    if not success:
        break

    # 밝기와 대비 조정
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # YOLO 예측 및 객체 추적
    results = model.track(frame, persist=True)
    
    # 결과가 없다면 빈 프레임을 그려서 출력
    if len(results) == 0 or results[0].boxes is None:
        # 감지된 객체가 없으면 빈 박스 또는 메시지 출력
        annotated_frame = frame
        print("No objects detected. Skipping this frame.")
    else:
        try:
            boxes = results[0].boxes.xywh.cpu()  # (x, y, w, h)
            track_ids = results[0].boxes.id.int().cpu().tolist()  # track id
            cls_ids = results[0].boxes.cls.int().cpu().tolist()  # 클래스 id
            names = results[0].names  # 클래스 이름
            annotated_frame = results[0].plot()  # 예측 결과 시각화

            # 손과 머리 추적 객체 저장
            hand_positions = {}  # 손의 추적된 위치
            head_positions = {}  # 머리의 추적된 위치
        except AttributeError:
        # 예상치 못한 오류가 발생하면 처리
            print("Error occurred while processing the results.")
            annotated_frame = frame  # 오류 발생 시 원본 프레임 그대로 사용

        # 클래스 이름을 기반으로 필터링
        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            # 'head' 또는 'hand' 클래스만 추적
            if names[cls_id] not in class_names:  # 'head'와 'hand'만 필터링
                continue
            
            x, y, w, h = box
            # 객체의 위치를 기반으로 추적 ID를 고정하기 위해 (x, y, w, h)를 기준으로 ID 매핑
            
            # 추적된 경로에 해당 객체의 중심점을 추가
            track_history[track_id].append((float(x + w / 2), float(y + h / 2)))

            # 'head'나 'hand'에 해당하는 객체는 해당 위치 저장
            if names[cls_id] == 'hand':
                hand_positions[track_id] = (x + w / 2, y + h / 2)
            elif names[cls_id] == 'head':
                head_positions[track_id] = (x + w / 2, y + h / 2)

            # 객체의 이동 경로에서 속도를 추정
            if len(track_history[track_id]) > 1:
                prev_position = track_history[track_id][-2]
                current_position = track_history[track_id][-1]
                # 두 지점 간 속도 계산 (단순히 두 점 사이의 차이)
                vx = current_position[0] - prev_position[0]
                vy = current_position[1] - prev_position[1]
                velocity = (vx, vy)
                
                # 예측된 위치 계산 (다음 프레임에서 위치 예측)
                predicted_position = predict_position(current_position, velocity)
                cv2.circle(annotated_frame, (int(predicted_position[0]), int(predicted_position[1])), 5, (0, 0, 255), -1)  # 빨간색 점

        # 손과 머리의 점들을 이어주는 선을 그리기 및 거리 출력
        for hand_id, hand_position in hand_positions.items():
            for head_id, head_position in head_positions.items():
                # 손과 머리의 중심 점을 이어주는 빨간색 선 그리기
                cv2.line(annotated_frame, 
                         (int(hand_position[0]), int(hand_position[1])), 
                         (int(head_position[0]), int(head_position[1])), 
                         (0, 0, 255), 2)  # 빨간색, 두께 2

                # 두 점 사이의 거리 계산
                distance = calculate_distance(hand_position, head_position)

                # 거리 출력 (거리 420px 이상일 때만 표시)
                if distance >= 420:
                    distance_text = f"Distance: {distance:.2f} px"
                    cv2.putText(annotated_frame, distance_text, 
                                (int((hand_position[0] + head_position[0]) / 2), 
                                 int((hand_position[1] + head_position[1]) / 2)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 추적 경로에 점을 계속 찍기 (모든 객체에 대해 이전 점도 계속 남도록)
        for track_id, track in track_history.items():
            # 경로를 포물선처럼 연결 (점들을 이어서 선을 그림)
            for i in range(1, len(track)):
                start_point = (int(track[i - 1][0]), int(track[i - 1][1]))
                end_point = (int(track[i][0]), int(track[i][1]))
                cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)  # 선 색상은 초록색

            # 각 점에 원을 그려서 포물선의 각 지점도 강조
            for point in track:
                cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # 파란색 점

    # 화면 크기 640x480으로 리사이즈
    resized_frame = cv2.resize(annotated_frame, (640, 640))

    # 결과를 화면에 표시
    cv2.imshow("Video Playback", resized_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 비디오 자원 해제 및 모든 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()
