from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from ultralytics import YOLO
from config.settings import settings
from schemas.observation import BoundingBox, VehicleObservation
from schemas.track import TrackRecord, TrackState
from tracking.track_manager import TrackManager

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class VehicleTracker:
    """
    High-level Vehicle Tracker integrating:
    1. YOLO detection + ByteTrack / BoT-SORT multi-object association
    2. TrackManager (Lifecycle State Machine: NEW -> ACTIVE -> TEMPORARILY_LOST -> ENDED)
    3. Consolidated camera-level vehicle observations
    """
    def __init__(
        self,
        camera_id: str = settings.CAMERA_ID,
        model_path: str = settings.YOLO_MODEL,
        conf_threshold: float = settings.CONF_THRESHOLD,
        min_hits: int = settings.MIN_HITS_TO_CONFIRM,
        max_lost_frames: int = settings.MAX_LOST_FRAMES,
        min_box_area: int = settings.MIN_BOX_AREA,
        tracker_config: str = settings.TRACKER_CONFIG,
        identity_pipeline: Optional[Any] = None
    ):
        self.camera_id = camera_id
        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config
        self.identity_pipeline = identity_pipeline
        
        print(f"[INFO] Initializing Vehicle Tracker ({tracker_config}) for camera: {camera_id}")
        self.model = YOLO(model_path)
        
        # Initialize lifecycle manager
        self.track_manager = TrackManager(
            camera_id=camera_id,
            min_hits_to_confirm=min_hits,
            max_lost_frames=max_lost_frames,
            min_box_area=min_box_area,
            identity_pipeline=identity_pipeline
        )

    def update(
        self,
        frame: np.ndarray,
        frame_num: int = 0,
        timestamp_sec: float = 0.0
    ) -> List[TrackRecord]:
        """
        Runs YOLO + BoT-SORT/ByteTrack, extracts raw associations,
        and delegates to TrackManager for state machine handling.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]

        detections = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()
            class_ids = results.boxes.cls.int().cpu().tolist()
            confidences = results.boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                detections.append({
                    "track_id": track_id,
                    "cls_id": cls_id,
                    "vehicle_type": VEHICLE_CLASSES.get(cls_id, "car"),
                    "conf": float(conf),
                    "bbox": (box[0], box[1], box[2], box[3])
                })

        # Update lifecycle state machine
        active_tracks = self.track_manager.update(
            detections=detections,
            frame=frame,
            frame_num=frame_num,
            timestamp_sec=timestamp_sec
        )
        return active_tracks

    def to_observation(self, record: TrackRecord) -> VehicleObservation:
        """Contract for Person 2 (ANPR & Re-ID)."""
        return self.track_manager.to_observation(record)
