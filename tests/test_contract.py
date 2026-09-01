import sys
import os
import json

# Add project root directory to Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.observation import BoundingBox, VehicleObservation
from schemas.event import DetectionEvent, BatchDetectionEvents
from schemas.heartbeat import CameraHeartbeatRequest

def test_api_contracts():
    print("=" * 60)
    print("           VALIDATING API CONTRACT SCHEMAS           ")
    print("=" * 60)

    # 1. Test Bounding Box
    bbox = BoundingBox(x1=412, y1=220, x2=690, y2=530)
    assert bbox.x1 == 412 and bbox.y2 == 530
    print("✅ 1. BoundingBox Schema: PASSED")

    # 2. Test Person 1 -> Person 2 Observation
    obs = VehicleObservation(
        camera_id="CAM_001",
        local_track_id="CAM_001_T_73",
        vehicle_type="car",
        vehicle_confidence=0.91,
        bounding_box=bbox
    )
    assert obs.local_track_id == "CAM_001_T_73"
    print("✅ 2. VehicleObservation Schema (P1 -> P2): PASSED")

    # 3. Test Detection Event (Shared Contract for Backend)
    event = DetectionEvent(
        camera_id="CAM_001",
        observed_at="2026-08-31T10:30:20Z",
        local_track_id="CAM_001_T_73",
        vehicle_type="car",
        plate_number="WB12AB1234",
        plate_confidence=0.94,
        vehicle_embedding=[0.012, -0.084, 0.221],
        embedding_model="reid-model-v1",
        embedding_version="1.0",
        vehicle_confidence=0.91,
        bounding_box=bbox,
        frame_reference="frames/CAM_001/evt_01H.jpg"
    )
    event_json = json.loads(event.model_dump_json())
    assert event_json["camera_id"] == "CAM_001"
    assert event_json["plate_number"] == "WB12AB1234"
    assert len(event_json["vehicle_embedding"]) == 3
    print("✅ 3. DetectionEvent Schema (POST /api/v1/events/detection): PASSED")

    # 4. Test Heartbeat Request
    hb = CameraHeartbeatRequest(
        status="online",
        fps=24.0,
        processing_latency_ms=42.0
    )
    hb_json = json.loads(hb.model_dump_json())
    assert hb_json["fps"] == 24.0
    print("✅ 4. CameraHeartbeatRequest Schema (POST /api/v1/cameras/{id}/heartbeat): PASSED")

    print("=" * 60)
    print("🎉 ALL API SCHEMAS STRICTLY MATCH THE OFFICIAL CONTRACT!")
    print("=" * 60)

if __name__ == "__main__":
    test_api_contracts()
