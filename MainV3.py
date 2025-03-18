import cv2
import numpy as np
from collections import defaultdict
from Object_Tracking import ObjectTracker  # YOLO 객체 추적 클래스
from Filter import ParticleFilter, KalmanFilter, HybridFilter  # 입자 필터, 칼만 필터

class Main:
    def __init__(self, model_path, video_path, num_particles=1000, process_noise=5, measurement_noise=10):
        self.model_path = model_path
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.track_history = defaultdict(list)
        self.tracker = ObjectTracker(self.model_path)

        # 손과 머리 위치를 저장하는 사전 (track_id 기반)
        self.hand_positions_history = defaultdict(list)
        self.head_positions_history = defaultdict(list)

        # 이전 프레임에서의 track_id와 현재 track_id를 매칭하는 사전
        self.previous_track_ids = {}

        # track_id와 class_id를 매핑할 사전
        self.track_class_map = {}

        # 하이브리드 필터 초기화
        initial_state = np.zeros(4)  # 초기 상태 [x, y, vx, vy]
        self.hybrid_filter = HybridFilter(num_particles, process_noise, measurement_noise, initial_state)

    def run(self):
        if not self.cap.isOpened():
            print("Error: 비디오 파일을 열 수 없습니다.")
            return

        # 창과 트랙바 설정
        cv2.namedWindow("Video Playback")
        cv2.createTrackbar("Brightness", "Video Playback", 100, 200, lambda x: None)  # 밝기 조절
        cv2.createTrackbar("Contrast", "Video Playback", 50, 100, lambda x: None)  # 대비 조절

        while self.cap.isOpened():
            # 트랙바 값 가져오기
            brightness = cv2.getTrackbarPos("Brightness", "Video Playback") - 100
            contrast = cv2.getTrackbarPos("Contrast", "Video Playback") / 50.0

            success, frame = self.cap.read()
            if not success:
                break

            # 밝기와 대비 적용
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

            boxes, track_ids, cls_ids = self.tracker.track_objects(frame)
            hand_positions = {}
            head_positions = {}

            # 손과 머리의 위치 업데이트
            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x, y, w, h = box
                center_x, center_y = x + w / 2, y + h / 2  # bbox의 중심

                # track_id와 class_id를 매핑하는 부분
                if cls_id == 0:  # 'hand' 클래스만 처리
                    if cls_id not in self.track_class_map:
                        self.track_class_map[cls_id] = track_id
                    hand_positions[track_id] = (center_x, center_y)
                elif cls_id == 1:  # 'head' 클래스만 처리
                    if cls_id not in self.track_class_map:
                        self.track_class_map[cls_id] = track_id
                    head_positions[track_id] = (center_x, center_y)

                # 필터 적용: 하이브리드 필터
                if track_id in hand_positions:
                    self.hybrid_filter.update((center_x, center_y))  # 하이브리드 필터 업데이트

                # bbox 그리기 (손과 머리)
                if cls_id == 0:  # 'hand'에 대한 bbox
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)  # 파란색 사각형
                elif cls_id == 1:  # 'head'에 대한 bbox
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)  # 초록색 사각형

            # 지나간 자리를 점으로 표시하기
            for hand_id, hand_position in hand_positions.items():
                self.hand_positions_history[hand_id].append(hand_position)
                # 지나간 손의 자리에 점을 찍기
                for pos in self.hand_positions_history[hand_id]:
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (255, 0, 0), -1)  # 파란색 점

            for head_id, head_position in head_positions.items():
                self.head_positions_history[head_id].append(head_position)
                # 지나간 머리의 자리에 점을 찍기
                for pos in self.head_positions_history[head_id]:
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)  # 초록색 점

            # 예측된 위치에 점을 찍기 (하이브리드 필터 예측 위치)
            for hand_id in hand_positions.keys():
                # 하이브리드 필터로 예측된 위치
                hybrid_estimate = self.hybrid_filter.resample()
                cv2.circle(frame, (int(hybrid_estimate[0]), int(hybrid_estimate[1])), 5, (0, 0, 255), -1)  # 빨간색 점

            # 사라진 객체의 경로도 그리기
            for hand_id, history in self.hand_positions_history.items():
                if hand_id not in hand_positions:
                    for pos in history:
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (255, 0, 0), -1)  # 파란색 점

            for head_id, history in self.head_positions_history.items():
                if head_id not in head_positions:
                    for pos in history:
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)  # 초록색 점

            # track_id 업데이트 (class_id와 track_id를 매핑하여 동일한 track_id 유지)
            self.previous_track_ids = dict(zip(track_ids, track_ids))

            # 화면 크기 640x640으로 리사이즈 후 표시
            resized_frame = cv2.resize(frame, (640, 640))
            cv2.imshow("Video Playback", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 자원 해제 및 창 닫기
        self.cap.release()
        cv2.destroyAllWindows()