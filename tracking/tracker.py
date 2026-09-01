from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
from config.settings import settings
from schemas.observation import BoundingBox, VehicleObservation

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

@dataclass
class TrackedVehicle:
    track_id: int
    local_track_id: str
    camera_id: str
    vehicle_type: str
    confidence: float
    bbox: BoundingBox
    center_point: Tuple[int, int]
    frame_crop: np.ndarray
    hits: int
    is_confirmed: bool

class VehicleTracker:
    """
    Multi-Object Vehicle Tracker powered by ByteTrack.
    Maintains persistent track IDs, filters false positive flickers,
    and produces VehicleObservation instances.
    """
    def __init__(
        self,
        camera_id: str = settings.CAMERA_ID,
        model_path: str = settings.YOLO_MODEL,
        conf_threshold: float = settings.CONF_THRESHOLD,
        min_hits: int = settings.MIN_HITS_TO_CONFIRM,
        min_box_area: int = settings.MIN_BOX_AREA,
        tracker_config: str = settings.TRACKER_CONFIG
    ):
        self.camera_id = camera_id
        self.min_hits = min_hits
        self.min_box_area = min_box_area
        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config

        print(f"[INFO] Initializing Vehicle Tracker for camera: {camera_id}")
        self.model = YOLO(model_path)
        
        # Track hit counts and motion history
        self.track_hits: Dict[int, int] = defaultdict(int)
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.reported_tracks: set = set()  # Track IDs already reported to Backend

    def update(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Processes a video frame through ByteTrack and returns a list of active confirmed vehicles.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]

        active_vehicles: List[TrackedVehicle] = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()
            class_ids = results.boxes.cls.int().cpu().tolist()
            confidences = results.boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = box
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                # Filter: Ignore tiny distant noise
                if (width * height) < self.min_box_area:
                    continue

                self.track_hits[track_id] += 1
                is_confirmed = self.track_hits[track_id] >= self.min_hits

                # Bottom-center road contact point
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)
                self.track_history[track_id].append((center_x, center_y))
                if len(self.track_history[track_id]) > 40:
                    self.track_history[track_id].pop(0)

                # Safe image crop of the vehicle for Person 2's ANPR model
                h_img, w_img = frame.shape[:2]
                crop_y1, crop_y2 = max(0, y1), min(h_img, y2)
                crop_x1, crop_x2 = max(0, x1), min(w_img, x2)
                frame_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                local_track_id = f"{self.camera_id}_T_{track_id}"
                vehicle_type = VEHICLE_CLASSES.get(cls_id, "vehicle")

                tracked = TrackedVehicle(
                    track_id=track_id,
                    local_track_id=local_track_id,
                    camera_id=self.camera_id,
                    vehicle_type=vehicle_type,
                    confidence=round(conf, 3),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    center_point=(center_x, center_y),
                    frame_crop=frame_crop,
                    hits=self.track_hits[track_id],
                    is_confirmed=is_confirmed
                )

                if is_confirmed:
                    active_vehicles.append(tracked)

        return active_vehicles

    def to_observation(self, vehicle: TrackedVehicle) -> VehicleObservation:
        """
        Converts a TrackedVehicle into the Person 1 -> Person 2 VehicleObservation contract.
        """
        return VehicleObservation(
            camera_id=vehicle.camera_id,
            local_track_id=vehicle.local_track_id,
            vehicle_type=vehicle.vehicle_type,
            vehicle_confidence=vehicle.confidence,
            bounding_box=vehicle.bbox,
            frame_crop=vehicle.frame_crop
        )
