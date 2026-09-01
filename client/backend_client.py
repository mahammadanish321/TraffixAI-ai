import time
import httpx
from typing import Optional
from config.settings import settings
from schemas.event import DetectionEvent, BatchDetectionEvents
from schemas.heartbeat import CameraHeartbeatRequest

class BackendClient:
    """
    HTTP Client communicating with Person 3's FastAPI Backend.
    Handles:
    - POST /api/v1/events/detection
    - POST /api/v1/events/batch
    - POST /api/v1/cameras/{camera_id}/heartbeat
    """
    def __init__(self, base_url: str = settings.BACKEND_URL, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        self.is_connected = False
        print(f"[INFO] Backend Client initialized with target URL: {self.base_url}")

    def send_detection_event(self, event: DetectionEvent) -> bool:
        """
        Sends a single representative vehicle observation to the Backend.
        Contract: POST /api/v1/events/detection
        """
        endpoint = settings.EVENT_ENDPOINT
        payload = event.model_dump(exclude={"frame_crop"})

        try:
            response = self.client.post(endpoint, json=payload)
            if response.status_code in (200, 201):
                return True
            else:
                print(f"[WARN] Backend rejected event ({response.status_code}): {response.text}")
                return False
        except httpx.RequestError:
            # Backend may be offline during standalone AI testing
            return False

    def send_heartbeat(self, camera_id: str, fps: float, processing_latency_ms: float) -> bool:
        """
        Sends periodic camera health heartbeat to Backend.
        Contract: POST /api/v1/cameras/{camera_id}/heartbeat
        """
        endpoint = settings.HEARTBEAT_ENDPOINT.format(camera_id=camera_id)
        payload = CameraHeartbeatRequest(
            status="online",
            fps=round(fps, 1),
            processing_latency_ms=round(processing_latency_ms, 1)
        ).model_dump()

        try:
            response = self.client.post(endpoint, json=payload)
            if response.status_code in (200, 201):
                self.is_connected = True
                return True
            else:
                self.is_connected = False
                return False
        except httpx.RequestError:
            self.is_connected = False
            return False

    def close(self):
        self.client.close()
