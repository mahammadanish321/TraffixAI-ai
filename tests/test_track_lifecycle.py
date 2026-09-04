import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracking.track_manager import TrackManager
from schemas.track import TrackState

def test_track_lifecycle_state_machine():
    print("=" * 65)
    print("       TESTING TRACK LIFECYCLE STATE MACHINE (TrackManager)       ")
    print("=" * 65)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    manager = TrackManager(
        camera_id="CAM_TEST",
        min_hits_to_confirm=3,
        max_lost_frames=5,
        min_box_area=100
    )

    # -------------------------------------------------------------
    # 1. Frame 1: Initial Detection -> State must be NEW
    # -------------------------------------------------------------
    det_frame_1 = [{
        "track_id": 1,
        "cls_id": 2,
        "vehicle_type": "car",
        "conf": 0.85,
        "bbox": (100, 100, 200, 200)
    }]
    active = manager.update(det_frame_1, dummy_frame, frame_num=1, timestamp_sec=0.1)
    track_1 = manager.tracks[1]
    assert track_1.state == TrackState.NEW, f"Expected NEW, got {track_1.state}"
    assert track_1.hits == 1
    assert len(manager.get_dispatchable_events()) == 0, "No event should be dispatched while NEW"
    print("✅ 1. Initial detection -> NEW state: PASSED")

    # -------------------------------------------------------------
    # 2. Frame 2: Second hit -> Still NEW (hits=2 < min_hits=3)
    # -------------------------------------------------------------
    det_frame_2 = [{
        "track_id": 1,
        "cls_id": 2,
        "vehicle_type": "car",
        "conf": 0.88,
        "bbox": (105, 100, 205, 200)
    }]
    manager.update(det_frame_2, dummy_frame, frame_num=2, timestamp_sec=0.2)
    assert track_1.state == TrackState.NEW
    assert track_1.hits == 2
    print("✅ 2. Second detection -> Remains NEW (unconfirmed): PASSED")

    # -------------------------------------------------------------
    # 3. Frame 3: Third hit -> Transition to ACTIVE & Event Queued
    # -------------------------------------------------------------
    det_frame_3 = [{
        "track_id": 1,
        "cls_id": 2,
        "vehicle_type": "car",
        "conf": 0.90,
        "bbox": (110, 100, 210, 200)
    }]
    active = manager.update(det_frame_3, dummy_frame, frame_num=3, timestamp_sec=0.3)
    assert track_1.state == TrackState.ACTIVE, f"Expected ACTIVE, got {track_1.state}"
    assert len(active) == 1
    events = manager.get_dispatchable_events()
    assert len(events) == 1, f"Expected 1 consolidated event, got {len(events)}"
    assert events[0].local_track_id == "CAM_TEST_T_1"
    assert events[0].vehicle_type == "car"
    print("✅ 3. Third detection (hits=3) -> Transition to ACTIVE & Event Generated: PASSED")

    # -------------------------------------------------------------
    # 4. Frame 4 & 5: Detection Missing -> Transition to TEMPORARILY_LOST & Coasting
    # -------------------------------------------------------------
    manager.update([], dummy_frame, frame_num=4, timestamp_sec=0.4)
    assert track_1.state == TrackState.TEMPORARILY_LOST, f"Expected TEMPORARILY_LOST, got {track_1.state}"
    assert track_1.missed_frame_count == 1
    # Check that position coasted forward with velocity
    coasted_x1 = track_1.bbox.x1
    assert coasted_x1 > 110, f"Coasting should have moved box forward, got {coasted_x1}"

    manager.update([], dummy_frame, frame_num=5, timestamp_sec=0.5)
    assert track_1.state == TrackState.TEMPORARILY_LOST
    assert track_1.missed_frame_count == 2
    print("✅ 4. Dropped frames -> Transition to TEMPORARILY_LOST & Velocity Coasting: PASSED")

    # -------------------------------------------------------------
    # 5. Frame 6: Detection Reappears -> RECOVER to ACTIVE (Same ID!)
    # -------------------------------------------------------------
    det_frame_6 = [{
        "track_id": 1,
        "cls_id": 2,
        "vehicle_type": "car",
        "conf": 0.89,
        "bbox": (125, 100, 225, 200)
    }]
    manager.update(det_frame_6, dummy_frame, frame_num=6, timestamp_sec=0.6)
    assert track_1.state == TrackState.ACTIVE, f"Expected ACTIVE on recovery, got {track_1.state}"
    assert track_1.missed_frame_count == 0
    assert track_1.local_track_id == "CAM_TEST_T_1", "Local track ID must remain unchanged"
    assert len(manager.get_dispatchable_events()) == 0, "No duplicate event on recovery"
    print("✅ 5. Re-detected vehicle -> Seamless RECOVERY to ACTIVE with same ID: PASSED")

    # -------------------------------------------------------------
    # 6. Frames 7 to 13: Vehicle disappears permanently -> Transition to ENDED
    # -------------------------------------------------------------
    for f in range(7, 14):
        manager.update([], dummy_frame, frame_num=f, timestamp_sec=f * 0.1)

    assert track_1.state == TrackState.ENDED, f"Expected ENDED after timeout, got {track_1.state}"
    assert len(manager.get_active_tracks()) == 0, "ENDED tracks must not be in active list"
    print("✅ 6. Exceeded max_lost_frames timeout -> Transition to ENDED: PASSED")

    print("=" * 65)
    print("🎉 ALL TRACK LIFECYCLE STATE MACHINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_track_lifecycle_state_machine()
