from pydantic import BaseModel, Field

class CameraHeartbeatRequest(BaseModel):
    """
    Official API Contract for: POST /api/v1/cameras/{camera_id}/heartbeat
    Reports that the AI worker is alive and operating normally.
    """
    status: str = "online"
    fps: float = Field(..., description="Current processing frames per second")
    processing_latency_ms: float = Field(..., description="Inference and tracking latency in milliseconds")

class CameraHeartbeatResponse(BaseModel):
    camera_id: str
    status: str
    last_seen: str
