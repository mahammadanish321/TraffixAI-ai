import cv2
import time
import sys
import os
import argparse
from typing import Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import settings
from camera.stream import VideoStream
from tracking.tracker import VehicleTracker
from schemas.track import TrackState
from client.backend_client import BackendClient
from identity import IdentityPipeline

def run_ai_service(
    camera_id: str = None,
    video_source: str = None,
    backend_url: str = None,
    headless: bool = False,
    loop: bool = False,
    max_frames: int = 0
):
    cam_id = camera_id or settings.CAMERA_ID
    src = video_source or settings.VIDEO_SOURCE
    backend = backend_url or settings.BACKEND_URL

    print("=" * 65)
    print(f"       TRAFFIX AI — AI SERVICE PIPELINE (CAMERA: {cam_id})       ")
    print("=" * 65)
    print("Architecture: YOLO (detect) -> BoT-SORT (track) -> Identity (ANPR + ReID)")
    print("Track Lifecycle: NEW -> ACTIVE -> TEMPORARILY_LOST -> ENDED")
    print("=" * 65)

    # 1. Initialize Pipeline Modules
    stream = VideoStream(source=src, loop=loop)
    identity_pipeline = IdentityPipeline()
    tracker = VehicleTracker(camera_id=cam_id, identity_pipeline=identity_pipeline)
    backend_client = BackendClient(base_url=backend)

    last_heartbeat_time = 0.0
    start_time = time.time()
    total_events_dispatched = 0

    print(f"[INFO] Streaming started: {src}")
    print(f"[INFO] Target Backend   : {backend}")
    print(f"[INFO] Headless Mode    : {headless}")
    if not headless:
        print("Press 'q' in the video window to stop.")
    print("=" * 65)

    try:
        for frame, frame_num, timestamp_sec in stream.read_frames():
            iter_start = time.time()

            # 2. Update Multi-Object Tracking & Lifecycle State Machine
            active_tracks = tracker.update(frame, frame_num=frame_num, timestamp_sec=timestamp_sec)

            # 3. Process and Dispatch Consolidated Events (No per-frame spam!)
            dispatchable_events = tracker.track_manager.get_dispatchable_events()
            for event in dispatchable_events:
                success = backend_client.send_detection_event(event)
                total_events_dispatched += 1
                status_str = "Dispatched to Backend" if success else "Generated (Backend Offline)"
                plate_display = f" | Plate: {event.plate_number}" if event.plate_number and not event.plate_number.startswith("UNREADABLE") else ""
                emb_display = f" | Emb: {len(event.vehicle_embedding)}D" if event.vehicle_embedding else ""
                print(f"🚗 [CONSOLIDATED EVENT] {event.local_track_id} ({event.vehicle_type}, conf={event.vehicle_confidence:.2f}{plate_display}{emb_display}) -> {status_str}")

            # 4. Render Visual Overlays on Frame
            for track in active_tracks:
                x1, y1, x2, y2 = track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2

                # Color-code by state:
                # Green = ACTIVE (directly detected)
                # Cyan = TEMPORARILY_LOST (coasting through occlusion)
                if track.state == TrackState.ACTIVE:
                    box_color = (0, 255, 128)
                    state_tag = f"{track.latest_confidence:.2f}"
                else:
                    box_color = (255, 255, 0)
                    state_tag = f"LOST (missed={track.missed_frame_count})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Display Plate number when identified, otherwise vehicle type
                if track.plate_number and not track.plate_number.startswith("UNREADABLE"):
                    label = f"{track.local_track_id} | {track.plate_number} [{track.plate_confidence:.2f}]"
                else:
                    label = f"{track.local_track_id} | {track.vehicle_type} [{state_tag}]"
                cv2.putText(frame, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

                # Draw motion trajectory trail
                trail = track.trajectory
                for i in range(1, len(trail)):
                    pt1 = (trail[i - 1][0], trail[i - 1][1])
                    pt2 = (trail[i][0], trail[i][1])
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            # 5. Measure latency & FPS
            iter_latency_ms = (time.time() - iter_start) * 1000
            elapsed = time.time() - start_time
            current_fps = frame_num / elapsed if elapsed > 0 else 0

            # 6. Periodic Camera Heartbeat
            current_now = time.time()
            if current_now - last_heartbeat_time >= settings.HEARTBEAT_INTERVAL_SEC:
                backend_client.send_heartbeat(
                    camera_id=cam_id,
                    fps=current_fps,
                    processing_latency_ms=iter_latency_ms
                )
                last_heartbeat_time = current_now

            # 7. Render HUD Overlay
            active_count = sum(1 for t in active_tracks if t.state == TrackState.ACTIVE)
            lost_count = sum(1 for t in active_tracks if t.state == TrackState.TEMPORARILY_LOST)
            hud_text = f"CAM: {cam_id} | FPS: {current_fps:.1f} | Active: {active_count} | Lost: {lost_count} | Events: {total_events_dispatched}"
            cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            if not headless:
                try:
                    cv2.imshow("Traffix AI — Track Lifecycle Pipeline", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] AI Service stopped by user.")
                        break
                except cv2.error:
                    pass

            if max_frames > 0 and frame_num >= max_frames:
                print(f"\n[INFO] Reached max-frames limit ({max_frames}). Finishing run.")
                break

    finally:
        stream.release()
        backend_client.close()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"[SUCCESS] AI Service session ended. Total consolidated events: {total_events_dispatched}")
        return total_events_dispatched

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffix AI — Multi-Camera Video Pipeline")
    parser.add_argument("--camera-id", type=str, default=settings.CAMERA_ID, help="Camera ID (e.g. CAM_001, CAM_002)")
    parser.add_argument("--video", type=str, default=settings.VIDEO_SOURCE, help="Path to video file or stream URL")
    parser.add_argument("--backend-url", type=str, default=settings.BACKEND_URL, help="Backend URL")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display (cv2.imshow)")
    parser.add_argument("--loop", action="store_true", help="Continuously loop the video file")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process before exiting (0 = all)")
    args = parser.parse_args()

    run_ai_service(
        camera_id=args.camera_id,
        video_source=args.video,
        backend_url=args.backend_url,
        headless=args.headless,
        loop=args.loop,
        max_frames=args.max_frames
    )
