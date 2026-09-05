import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()

@dataclass
class Settings:
    # Camera Identification & GPS Geolocation
    CAMERA_ID: str = os.getenv("CAMERA_ID", "CAM_001")
    CAMERA_NAME: str = os.getenv("CAMERA_NAME", "Junction A")
    CAMERA_LATITUDE: float = float(os.getenv("CAMERA_LATITUDE", "22.5726"))
    CAMERA_LONGITUDE: float = float(os.getenv("CAMERA_LONGITUDE", "88.3639"))
    
    # Video Source (file path, webcam index '0', or RTSP URL)
    VIDEO_SOURCE: str = os.getenv("VIDEO_SOURCE", "data/videos/sample_traffic.mp4")
    
    # YOLO & Tracking Settings
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    CONF_THRESHOLD: float = float(os.getenv("CONF_THRESHOLD", "0.20"))
    TRACKER_CONFIG: str = os.getenv("TRACKER_CONFIG", "config/tracker.yaml")
    MIN_HITS_TO_CONFIRM: int = int(os.getenv("MIN_HITS_TO_CONFIRM", "4"))
    MIN_BOX_AREA: int = int(os.getenv("MIN_BOX_AREA", "625"))  # 25x25 pixels
    MAX_LOST_FRAMES: int = int(os.getenv("MAX_LOST_FRAMES", "20"))  # Tolerance for temporary occlusion before ENDED
    
    # Re-ID Model Settings (Person 2)
    REID_MODEL_NAME: str = os.getenv("REID_MODEL_NAME", "mobilenet_v3_small")
    REID_EMBEDDING_DIM: int = int(os.getenv("REID_EMBEDDING_DIM", "512"))
    REID_SIMILARITY_THRESHOLD: float = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.75"))

    # Backend API Endpoints (Person 3)
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    EVENT_ENDPOINT: str = "/api/v1/events/detection"
    BATCH_ENDPOINT: str = "/api/v1/events/batch"
    HEARTBEAT_ENDPOINT: str = "/api/v1/cameras/{camera_id}/heartbeat"
    
    # Heartbeat interval in seconds
    HEARTBEAT_INTERVAL_SEC: int = int(os.getenv("HEARTBEAT_INTERVAL_SEC", "15"))

# Global singleton settings instance
settings = Settings()
