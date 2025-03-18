# Main.py
import cv2
import numpy as np
from collections import defaultdict
from Object_Tracking import ObjectTracker  # YOLO 객체 추적 클래스
from Filter import HybridFilter  # 입자 필터, 칼만 필터
from Utils import Trajectory, TrajectoryManager  # Utils.py에서 정의된 클래스 임포트

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

        # 트래젝토리 관리 객체 초기화
        self.trajectory_manager = TrajectoryManager()  # TrajectoryManager 객체 생성

    def run(self):
        if not self.cap.isOpened():
            print("Error: 비디오 파일을 열 수 없습니다.")
            return

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

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
                if track_id in head_positions:
                    self.hybrid_filter.update((center_x, center_y))  # 머리 위치도 필터 업데이트

                # 궤적 업데이트 (손과 머리 모두)
                if track_id in hand_positions or track_id in head_positions:
                    self.trajectory_manager.setPointsFrame([np.array([center_x, center_y])])  # Trajectory에 포인트 추가

            # 궤적을 화면에 그리기
            frame = drawTrajectory(frame, self.trajectory_manager)

            # 화면 크기 640x640으로 리사이즈 후 표시
            resized_frame = cv2.resize(frame, (640, 640))
            cv2.imshow("Video Playback", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 자원 해제 및 창 닫기
        self.cap.release()
        cv2.destroyAllWindows()


# 궤적을 화면에 그리는 함수
def drawTrajectory(frame, trajectory_manager):
    trajectories = trajectory_manager.getTrajectorys()
    if not trajectories:  # 궤적이 없다면 처리하지 않음
        return frame

    # 궤적이 있는 경우 그리기
    for traj in trajectories:
        points = traj.getPoints()
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i-1]))
            pt2 = tuple(map(int, points[i]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # 초록색 선으로 궤적 그리기
    return frame
