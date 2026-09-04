import cv2
import time
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from tracking.tracker import VehicleTracker
from schemas.track import TrackState
from camera.stream import VideoStream

def run_continuous_tracking_test(
    video_path: str = settings.VIDEO_SOURCE,
    camera_id: str = settings.CAMERA_ID
):
    """
    Test script for verified continuous vehicle tracking:
    - Verifies that vehicles maintain a single persistent local_track_id
    - Shows EMA bounding box smoothing (zero jitter)
    - Demonstrates coasting / extrapolation when detection drops
    - Reports track state transitions: ACTIVE vs TEMPORARILY_LOST
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    print("=" * 65)
    print(f"      CONTINUOUS VEHICLE TRACKING VERIFICATION ({camera_id})      ")
    print("=" * 65)
    print("Features Active: BoT-SORT + TrackManager (Lifecycle State Machine)")
    print("States: ACTIVE (Direct Detection) | TEMPORARILY_LOST (Coasting)")
    print("Press 'q' in the window to stop.")
    print("=" * 65)

    stream = VideoStream(source=video_path, loop=False)
    tracker = VehicleTracker(camera_id=camera_id)

    track_first_seen = {}
    track_last_seen = {}
    track_hit_counts = {}

    start_time = time.time()
    total_frames = 0

    for frame, frame_num, timestamp_sec in stream.read_frames():
        total_frames = frame_num
        active_tracks = tracker.update(frame, frame_num=frame_num, timestamp_sec=timestamp_sec)

        for track in active_tracks:
            tid = track.track_id
            if tid not in track_first_seen:
                track_first_seen[tid] = frame_num
            track_last_seen[tid] = frame_num
            track_hit_counts[tid] = track_hit_counts.get(tid, 0) + 1

            x1, y1, x2, y2 = track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2

            # Visual colors:
            # Green = Direct high-confidence ACTIVE detection
            # Cyan = TEMPORARILY_LOST / Coasting (preserving ID across momentary dropout)
            if track.state == TrackState.TEMPORARILY_LOST:
                box_color = (255, 255, 0)  # Cyan
                status_tag = f"[LOST: {track.missed_frame_count}f]"
            else:
                box_color = (0, 255, 128)  # Bright Green
                status_tag = f"{track.latest_confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            label = f"{track.local_track_id} | {track.vehicle_type} {status_tag}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA
            )

            # Draw trajectory trail
            trail = track.trajectory
            for i in range(1, len(trail)):
                pt1 = (trail[i - 1][0], trail[i - 1][1])
                pt2 = (trail[i][0], trail[i][1])
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        elapsed = time.time() - start_time
        fps = frame_num / elapsed if elapsed > 0 else 0

        hud = f"FPS: {fps:.1f} | Frame: {frame_num} | Active: {len(active_tracks)} | Unique Confirmed IDs: {len(track_first_seen)}"
        cv2.putText(frame, hud, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Traffix AI — Continuous Tracking Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print("           CONTINUOUS TRACK LIFETIME REPORT           ")
    print("=" * 65)
    print(f"Total Video Frames Processed : {total_frames}")
    print(f"Total Vehicles Tracked       : {len(track_first_seen)}")
    print("-" * 65)
    print(f"{'Local Track ID':<16} | {'Start Frame':<12} | {'End Frame':<10} | {'Duration (Frames)':<15}")
    print("-" * 65)
    for tid in sorted(track_first_seen.keys()):
        start_f = track_first_seen[tid]
        end_f = track_last_seen[tid]
        duration = end_f - start_f + 1
        print(f"CAM_001_T_{tid:<7} | {start_f:<12} | {end_f:<10} | {duration:<15}")
    print("=" * 65)

if __name__ == "__main__":
    run_continuous_tracking_test()
