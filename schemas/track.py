from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
from datetime import datetime, timezone
import numpy as np
from schemas.observation import BoundingBox

class TrackState(str, Enum):
    """
    Lifecycle states for an individual vehicle track:
    NEW: First seen or below confirmation hit threshold.
    ACTIVE: Confirmed track actively detected in current frame.
    TEMPORARILY_LOST: Missed in current frame, but within max_lost_frames tolerance (coasting).
    ENDED: Missing beyond timeout or exited camera view. Finalized for cleanup.
    """
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    ENDED = "ENDED"

@dataclass
class TrackRecord:
    """
    Complete in-memory state for a tracked vehicle.
    """
    track_id: int
    local_track_id: str
    camera_id: str
    vehicle_type: str
    state: TrackState = TrackState.NEW
    
    # Detection metrics
    latest_confidence: float = 0.0
    peak_confidence: float = 0.0
    
    # Coordinates & Motion
    bbox: BoundingBox = field(default_factory=lambda: BoundingBox(x1=0, y1=0, x2=0, y2=0))
    smoothed_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels/frame
    trajectory: List[Tuple[int, int, float]] = field(default_factory=list) # (center_x, center_y, timestamp)
    
    # Lifetime tracking
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    first_frame: int = 0
    last_frame: int = 0
    hits: int = 0
    missed_frame_count: int = 0
    
    # Observation & Enrichment for Person 2 (ANPR / Re-ID)
    best_crop: Optional[np.ndarray] = None
    best_crop_score: float = 0.0  # area * confidence
    plate_number: Optional[str] = None
    plate_confidence: Optional[float] = None
    vehicle_embedding: Optional[List[float]] = None
    
    # Event dispatch control
    event_dispatched: bool = False
    ended_event_dispatched: bool = False
