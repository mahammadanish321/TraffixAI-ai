from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone
import numpy as np
from config.settings import settings
from schemas.track import TrackState, TrackRecord
from schemas.observation import BoundingBox, VehicleObservation
from schemas.event import DetectionEvent

class TrackManager:
    """
    Vehicle Track Lifecycle Manager & State Machine.
    Manages track lifecycle:
        NEW -> ACTIVE -> TEMPORARILY_LOST -> ACTIVE (recovered) -> ENDED
    
    Responsibilities:
    - Maintains in-memory track states
    - Handles temporary detection drops via velocity coasting without dropping local ID
    - Selects the best vehicle image crop for Person 2 (ANPR & Re-ID)
    - Consolidates camera observations (prevents per-frame event spam)
    - Provides dispatchable DetectionEvents matching the official API contract
    """
    def __init__(
        self,
        camera_id: str = settings.CAMERA_ID,
        min_hits_to_confirm: int = settings.MIN_HITS_TO_CONFIRM,
        max_lost_frames: int = settings.MAX_LOST_FRAMES,
        min_box_area: int = settings.MIN_BOX_AREA,
        ema_alpha: float = 0.75,
        identity_pipeline: Optional[Any] = None
    ):
        self.camera_id = camera_id
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_lost_frames = max_lost_frames
        self.min_box_area = min_box_area
        self.ema_alpha = ema_alpha
        self.identity_pipeline = identity_pipeline

        # In-memory track registry: track_id -> TrackRecord
        self.tracks: Dict[int, TrackRecord] = {}
        
        # Internal queue of consolidated events ready to dispatch to Backend
        self._dispatch_queue: List[DetectionEvent] = []

    def update(
        self,
        detections: List[dict],
        frame: np.ndarray,
        frame_num: int,
        timestamp_sec: float
    ) -> List[TrackRecord]:
        """
        Updates the track lifecycle state machine with current frame detections.
        detections: List of dicts containing:
            {'track_id': int, 'cls_id': int, 'vehicle_type': str, 'conf': float, 'bbox': (x1, y1, x2, y2)}
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        current_detected_ids = set()
        h_img, w_img = frame.shape[:2]

        # 1. Process all detected vehicles in this frame
        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            vehicle_type = det["vehicle_type"]

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            box_area = width * height

            # Filter tiny distant noise
            if box_area < self.min_box_area:
                continue

            current_detected_ids.add(track_id)

            if track_id not in self.tracks:
                # ── State Transition: Spawn NEW track ──
                record = TrackRecord(
                    track_id=track_id,
                    local_track_id=f"{self.camera_id}_T_{track_id}",
                    camera_id=self.camera_id,
                    vehicle_type=vehicle_type,
                    state=TrackState.NEW,
                    latest_confidence=conf,
                    peak_confidence=conf,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, frame_width=w_img, frame_height=h_img),
                    smoothed_bbox=(float(x1), float(y1), float(x2), float(y2)),
                    velocity=(0.0, 0.0),
                    first_seen=now_iso,
                    last_seen=now_iso,
                    first_frame=frame_num,
                    last_frame=frame_num,
                    hits=1,
                    missed_frame_count=0
                )
                self.tracks[track_id] = record
            else:
                record = self.tracks[track_id]
                record.hits += 1
                record.missed_frame_count = 0
                record.last_seen = now_iso
                record.last_frame = frame_num
                record.latest_confidence = conf
                record.peak_confidence = max(record.peak_confidence, conf)
                record.vehicle_type = vehicle_type

                # ── State Transition: Check ACTIVE or RECOVERY ──
                if record.state == TrackState.NEW and record.hits >= self.min_hits_to_confirm:
                    record.state = TrackState.ACTIVE
                elif record.state == TrackState.TEMPORARILY_LOST:
                    # Successfully recovered without losing the track ID!
                    record.state = TrackState.ACTIVE

                # Smooth Bounding Box (EMA)
                prev_x1, prev_y1, prev_x2, prev_y2 = record.smoothed_bbox
                sm_x1 = self.ema_alpha * x1 + (1 - self.ema_alpha) * prev_x1
                sm_y1 = self.ema_alpha * y1 + (1 - self.ema_alpha) * prev_y1
                sm_x2 = self.ema_alpha * x2 + (1 - self.ema_alpha) * prev_x2
                sm_y2 = self.ema_alpha * y2 + (1 - self.ema_alpha) * prev_y2

                # Compute Velocity Vector
                prev_cx = (prev_x1 + prev_x2) / 2
                curr_cx = (sm_x1 + sm_x2) / 2
                prev_cy = prev_y2
                curr_cy = sm_y2
                record.velocity = (curr_cx - prev_cx, curr_cy - prev_cy)
                record.smoothed_bbox = (sm_x1, sm_y1, sm_x2, sm_y2)
                record.bbox = BoundingBox(x1=int(sm_x1), y1=int(sm_y1), x2=int(sm_x2), y2=int(sm_y2), frame_width=w_img, frame_height=h_img)

            # Record trajectory contact point (center bottom)
            center_x = int((record.smoothed_bbox[0] + record.smoothed_bbox[2]) / 2)
            center_y = int(record.smoothed_bbox[3])
            record.trajectory.append((center_x, center_y, timestamp_sec))
            if len(record.trajectory) > 60:
                record.trajectory.pop(0)

            # Evaluate Crop Quality for Person 2 (Area * Confidence)
            crop_score = box_area * conf
            is_better_crop = False
            if crop_score > record.best_crop_score:
                crop_y1, crop_y2 = max(0, int(record.smoothed_bbox[1])), min(h_img, int(record.smoothed_bbox[3]))
                crop_x1, crop_x2 = max(0, int(record.smoothed_bbox[0])), min(w_img, int(record.smoothed_bbox[2]))
                if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                    record.best_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    record.best_crop_score = crop_score
                    is_better_crop = True

            # Trigger consolidated event once confirmed ACTIVE or when a better plate is found
            if record.state == TrackState.ACTIVE:
                if not record.event_dispatched:
                    self._enqueue_event(record)
                    record.event_dispatched = True
                elif is_better_crop and self.identity_pipeline is not None and (not record.plate_number or record.plate_number.startswith("TRACK_")):
                    # Try reading plate on new improved crop
                    try:
                        p_num, p_conf, _ = self.identity_pipeline.extract_identity(record.best_crop)
                        if p_num and not p_num.startswith("UNREADABLE") and len(p_num) >= 4:
                            record.plate_number = p_num
                            record.plate_confidence = p_conf or 0.85
                            self._enqueue_event(record)
                    except Exception:
                        pass

        # 2. Process vehicles NOT detected in this frame (handle missing/coasting/ended)
        for track_id, record in list(self.tracks.items()):
            if track_id not in current_detected_ids:
                record.missed_frame_count += 1

                if record.state == TrackState.NEW:
                    # Unconfirmed noise that disappeared immediately -> discard
                    if record.missed_frame_count > 2:
                        del self.tracks[track_id]

                elif record.state == TrackState.ACTIVE:
                    # ── State Transition: ACTIVE -> TEMPORARILY_LOST ──
                    record.state = TrackState.TEMPORARILY_LOST
                    self._coast_track(record, w_img, h_img, timestamp_sec)

                elif record.state == TrackState.TEMPORARILY_LOST:
                    if record.missed_frame_count > self.max_lost_frames:
                        # ── State Transition: TEMPORARILY_LOST -> ENDED ──
                        record.state = TrackState.ENDED
                        if record.event_dispatched and not getattr(record, 'exit_dispatched', False):
                            record.exit_dispatched = True
                            self._enqueue_exit_event(record)
                    else:
                        # Continue coasting
                        self._coast_track(record, w_img, h_img, timestamp_sec)

        return self.get_active_tracks()

    def _enqueue_exit_event(self, record: TrackRecord):
        """Dispatches vehicle exited lifecycle notification to backend."""
        event = DetectionEvent(
            camera_id=record.camera_id,
            observed_at=record.last_seen,
            local_track_id=record.local_track_id,
            vehicle_type=record.vehicle_type,
            vehicle_confidence=record.peak_confidence,
            bounding_box=record.bbox,
            plate_number=record.plate_number,
            plate_confidence=record.plate_confidence,
            vehicle_embedding=[],
            status="exited"
        )
        self._dispatch_queue.append(event)

    def _coast_track(self, record: TrackRecord, w_img: int, h_img: int, timestamp_sec: float):
        """Extrapolates track position using last known velocity during temporary detection drops."""
        vx, vy = record.velocity
        bx1, by1, bx2, by2 = record.smoothed_bbox
        bx1 += vx
        bx2 += vx
        by1 += vy
        by2 += vy

        # Keep inside image bounds
        bx1 = max(0, min(w_img - 10, bx1))
        bx2 = max(10, min(w_img, bx2))
        by1 = max(0, min(h_img - 10, by1))
        by2 = max(10, min(h_img, by2))

        record.smoothed_bbox = (bx1, by1, bx2, by2)
        record.bbox = BoundingBox(x1=int(bx1), y1=int(by1), x2=int(bx2), y2=int(by2))

        center_x = int((bx1 + bx2) / 2)
        center_y = int(by2)
        record.trajectory.append((center_x, center_y, timestamp_sec))
        if len(record.trajectory) > 60:
            record.trajectory.pop(0)

    def _enqueue_event(self, record: TrackRecord):
        """Builds and queues a consolidated DetectionEvent for a confirmed vehicle."""
        if self.identity_pipeline is not None and record.best_crop is not None:
            obs = self.to_observation(record)
            event = self.identity_pipeline.enrich_observation(obs)
            # Store identity data into track record for HUD visualization
            record.plate_number = event.plate_number
            record.plate_confidence = event.plate_confidence
            record.vehicle_embedding = event.vehicle_embedding
        else:
            event = DetectionEvent(
                camera_id=record.camera_id,
                observed_at=record.last_seen,
                local_track_id=record.local_track_id,
                vehicle_type=record.vehicle_type,
                vehicle_confidence=record.peak_confidence,
                bounding_box=record.bbox,
                plate_number=record.plate_number or f"UNREADABLE_{record.local_track_id}",
                plate_confidence=record.plate_confidence or 0.0,
                vehicle_embedding=record.vehicle_embedding or [],
                embedding_model="mobilenet_v3_small",
                embedding_version="1.0",
                first_seen=record.first_seen,
                last_seen=record.last_seen
            )
        self._dispatch_queue.append(event)

    def get_dispatchable_events(self) -> List[DetectionEvent]:
        """Pops and returns all pending consolidated events."""
        events = list(self._dispatch_queue)
        self._dispatch_queue.clear()
        return events

    def get_active_tracks(self) -> List[TrackRecord]:
        """Returns all confirmed tracks that are ACTIVE or TEMPORARILY_LOST."""
        return [
            t for t in self.tracks.values()
            if t.state in (TrackState.ACTIVE, TrackState.TEMPORARILY_LOST)
        ]

    def to_observation(self, record: TrackRecord) -> VehicleObservation:
        """Contract for Person 2 (ANPR & Re-ID)."""
        return VehicleObservation(
            camera_id=record.camera_id,
            local_track_id=record.local_track_id,
            vehicle_type=record.vehicle_type,
            vehicle_confidence=record.peak_confidence,
            bounding_box=record.bbox,
            frame_crop=record.best_crop
        )

    def cleanup_ended_tracks(self, max_idle_frames: int = 150):
        """Cleans up tracks that have ended and been idle to prevent memory leaks."""
        current_frame = max((t.last_frame for t in self.tracks.values()), default=0)
        for tid in list(self.tracks.keys()):
            record = self.tracks[tid]
            if record.state == TrackState.ENDED and (current_frame - record.last_frame) > max_idle_frames:
                del self.tracks[tid]
