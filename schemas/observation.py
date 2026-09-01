from typing import Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: int = Field(..., description="Top-left X pixel coordinate")
    y1: int = Field(..., description="Top-left Y pixel coordinate")
    x2: int = Field(..., description="Bottom-right X pixel coordinate")
    y2: int = Field(..., description="Bottom-right Y pixel coordinate")

class VehicleObservation(BaseModel):
    """
    Contract between Person 1 (CV) and Person 2 (ANPR & Re-ID).
    Person 1 generates this observation for a tracked vehicle.
    """
    camera_id: str
    local_track_id: str
    observed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    vehicle_type: str  # car, motorcycle, bus, truck
    vehicle_confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    
    # Internal frame crop reference for Person 2's OCR/Re-ID model
    frame_crop: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True
