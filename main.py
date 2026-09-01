import cv2
import time
import sys
import os
from typing import Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import settings
from camera.stream import VideoStream
from tracking.tracker import VehicleTracker
from client.backend_client import BackendClient
from schemas.event import DetectionEvent

def run_ai_service():
    print("=" * 65)
    print(f"       TRAFFIX AI — AI SERVICE PIPELINE (CAMERA: {settings.CAMERA_ID})       ")
    print("=" * 65)

    # 1. Initialize Pipeline Modules
    stream = VideoStream(source=settings.VIDEO_SOURCE, loop=True)
    tracker = VehicleTracker(camera_id=settings.CAMERA_ID)
    backend_client = BackendClient(base_url=settings.BACKEND_URL)

    # Track IDs that have already had a representative event dispatched to Backend
    dispatched_tracks = set()

    last_heartbeat_time = 0.0
    start_time = time.time()
    total_events_sent = 0

    print(f"[INFO] Streaming started: {settings.VIDEO_SOURCE}")
    print(f"[INFO] Target Backend   : {settings.BACKEND_URL}")
    print("Press 'q' in the video window to stop.")
    print("=" * 65)

    try:
        for frame, frame_num, timestamp_sec in stream.read_frames():
            iter_start = time.time()

            # 2. Multi-Object Tracking & Vehicle Confirmation
            active_vehicles = tracker.update(frame)

            # 3. Process confirmed vehicle observations
            for vehicle in active_vehicles:
                # Dispatch 1 representative event per confirmed vehicle track
                if vehicle.track_id not in dispatched_tracks:
                    dispatched_tracks.add(vehicle.track_id)

                    # Build official DetectionEvent (Person 1 + P2 stub)
                    event = DetectionEvent(
                        camera_id=vehicle.camera_id,
                        local_track_id=vehicle.local_track_id,
                        vehicle_type=vehicle.vehicle_type,
                        vehicle_confidence=vehicle.confidence,
                        bounding_box=vehicle.bbox,
                        plate_number="TEMP_" + vehicle.local_track_id, # P2 ANPR will populate
                        plate_confidence=0.90,
                        vehicle_embedding=[0.012, -0.084, 0.221],       # P2 Re-ID will populate
                        embedding_model="reid-model-v1",
                        embedding_version="1.0"
                    )

                    success = backend_client.send_detection_event(event)
                    total_events_sent += 1
                    status_str = "Dispatched to Backend" if success else "Generated (Backend Offline)"
                    print(f"🚗 [EVENT] {vehicle.local_track_id} ({vehicle.vehicle_type}, {vehicle.confidence}) -> {status_str}")

                # Draw bounding box on frame
                x1, y1, x2, y2 = vehicle.bbox.x1, vehicle.bbox.y1, vehicle.bbox.x2, vehicle.bbox.y2
                box_color = (0, 255, 128) if vehicle.vehicle_type == "car" else (255, 160, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                label = f"{vehicle.local_track_id} | {vehicle.vehicle_type} {vehicle.confidence}"
                cv2.putText(frame, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

                # Draw motion trail
                trail = tracker.track_history[vehicle.track_id]
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), 2)

            # 4. Measure latency & FPS
            iter_latency_ms = (time.time() - iter_start) * 1000
            elapsed = time.time() - start_time
            current_fps = frame_num / elapsed if elapsed > 0 else 0

            # 5. Periodic Camera Heartbeat
            current_now = time.time()
            if current_now - last_heartbeat_time >= settings.HEARTBEAT_INTERVAL_SEC:
                hb_sent = backend_client.send_heartbeat(
                    camera_id=settings.CAMERA_ID,
                    fps=current_fps,
                    processing_latency_ms=iter_latency_ms
                )
                last_heartbeat_time = current_now

            # 6. Render HUD Overlay
            hud_text = f"CAM: {settings.CAMERA_ID} | FPS: {current_fps:.1f} | Latency: {iter_latency_ms:.1f}ms | Events: {total_events_sent}"
            cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Traffix AI — Production AI Service", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] AI Service stopped by user.")
                break

    finally:
        stream.release()
        backend_client.close()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] AI Service session ended. Total events processed: {total_events_sent}")

if __name__ == "__main__":
    run_ai_service()
