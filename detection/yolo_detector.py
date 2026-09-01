from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
from config.settings import settings
from schemas.observation import BoundingBox

# Road vehicle classes in MS COCO dataset
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

@dataclass
class DetectionResult:
    class_id: int
    vehicle_type: str
    confidence: float
    bbox: BoundingBox

class YOLODetector:
    """
    YOLO Vehicle Detection Engine.
    Filters frames specifically for road vehicles (car, bus, truck, motorcycle).
    """
    def __init__(
        self,
        model_path: str = settings.YOLO_MODEL,
        conf_threshold: float = settings.CONF_THRESHOLD
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        print(f"[INFO] Initializing YOLO Detector with weights: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Runs object detection on a single frame and returns road vehicle detections.
        """
        results = self.model(
            source=frame,
            conf=self.conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]

        detections: List[DetectionResult] = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            class_ids = results.boxes.cls.int().cpu().tolist()
            confidences = results.boxes.conf.cpu().tolist()

            for box, cls_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                detections.append(
                    DetectionResult(
                        class_id=cls_id,
                        vehicle_type=VEHICLE_CLASSES.get(cls_id, "vehicle"),
                        confidence=round(conf, 3),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    )
                )

        return detections
