import numpy as np
from typing import Optional, Tuple, List
import torch
from anpr import ANPREngine
from reid import VehicleReIDExtractor
from schemas.observation import VehicleObservation
from schemas.event import DetectionEvent

class IdentityPipeline:
    """
    Person 2 Identity Intelligence Orchestrator.
    
    Bridges Person 1's VehicleObservation (motion & localization)
    with Person 2's ANPR (text identity) and Re-ID (visual identity),
    producing a complete, enriched DetectionEvent for Person 3 (Backend).
    """
    def __init__(self, use_gpu: Optional[bool] = None):
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        print(f"[INFO] Initializing Person 2 IdentityPipeline (GPU={use_gpu})...")
        self.anpr = ANPREngine(gpu=use_gpu)
        self.reid = VehicleReIDExtractor(device="cuda" if use_gpu else "cpu")
        print("[SUCCESS] IdentityPipeline (ANPR + Re-ID) ready.")

    def extract_identity(
        self,
        vehicle_crop: np.ndarray
    ) -> Tuple[Optional[str], Optional[float], Optional[List[float]]]:
        """
        Extracts both license plate and visual feature embedding from a vehicle crop.
        Returns:
            (plate_number, plate_confidence, vehicle_embedding)
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return (None, None, None)

        # 1. ANPR: Extract and correct license plate
        plate_number, plate_confidence = self.anpr.read_plate(vehicle_crop)

        # 2. Re-ID: Extract 512-D normalized visual appearance embedding
        embedding = self.reid.extract(vehicle_crop)

        return (plate_number, plate_confidence, embedding)

    def enrich_observation(self, obs: VehicleObservation) -> DetectionEvent:
        """
        Takes a VehicleObservation from Person 1 and returns an official DetectionEvent.
        """
        plate_number, plate_confidence, embedding = None, None, None

        if obs.frame_crop is not None and obs.frame_crop.size > 0:
            plate_number, plate_confidence, embedding = self.extract_identity(obs.frame_crop)

        # Genuine license plate read by EasyOCR (or None if plate is unreadable)
        clean_plate = plate_number if (plate_number and not plate_number.startswith("UNREADABLE")) else None

        event = DetectionEvent(
            camera_id=obs.camera_id,
            observed_at=obs.observed_at,
            local_track_id=obs.local_track_id,
            vehicle_type=obs.vehicle_type,
            vehicle_confidence=obs.vehicle_confidence,
            bounding_box=obs.bounding_box,
            plate_number=clean_plate,
            plate_confidence=plate_confidence if plate_confidence is not None else 0.0,
            vehicle_embedding=embedding or [],
            embedding_model=self.reid.model_name,
            embedding_version=self.reid.model_version,
            first_seen=obs.observed_at,
            last_seen=obs.observed_at
        )
        return event
