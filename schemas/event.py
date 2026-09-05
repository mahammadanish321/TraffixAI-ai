from typing import List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field
from schemas.observation import BoundingBox

class DetectionEvent(BaseModel):
    """
    Official API Contract for: POST /api/v1/events/detection
    Sent from AI Service to Backend (Person 3).
    """
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    camera_id: str
    observed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    local_track_id: str
    vehicle_type: str
    plate_number: Optional[str] = None
    plate_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    vehicle_embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = "reid-model-v1"
    embedding_version: Optional[str] = "1.0"
    vehicle_confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    frame_reference: Optional[str] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


class BatchDetectionEvents(BaseModel):
    """
    Official API Contract for: POST /api/v1/events/batch
    """
    events: List[DetectionEvent]
