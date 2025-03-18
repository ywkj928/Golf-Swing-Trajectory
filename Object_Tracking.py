from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # YOLO 모델 로드

    def track_objects(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")  # YOLO 추적 결과
        boxes, track_ids, cls_ids = [], [], []

        if len(results) > 0 and results[0].boxes is not None:
            try:
                boxes = results[0].boxes.xywh.cpu()  # (x, y, w, h)
                track_ids = results[0].boxes.id.int().cpu().tolist()  # track id
                cls_ids = results[0].boxes.cls.int().cpu().tolist()  # 클래스 id
            except AttributeError as e:
                print(f"Error processing detection: {e}")

        return boxes, track_ids, cls_ids


