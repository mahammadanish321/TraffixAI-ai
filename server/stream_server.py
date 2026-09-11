import os
import sys
import time
import cv2
import numpy as np
import threading
from typing import Dict, Optional, Generator
from fastapi import FastAPI, Response, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO

# Ensure Traffix_Ai root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.settings import settings
from camera.stream import VideoStream
from tracking.tracker import VehicleTracker
from schemas.track import TrackState
from client.backend_client import BackendClient
from identity import IdentityPipeline

app = FastAPI(title="Traffix AI Live Stream Daemon", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and cache
plate_model: Optional[YOLO] = None
plate_model_lock = threading.Lock()

def get_plate_model() -> Optional[YOLO]:
    global plate_model
    with plate_model_lock:
        if plate_model is None:
            model_path = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
            if os.path.exists(model_path):
                print(f"[INFO] Loading License Plate Detector: {model_path}")
                try:
                    plate_model = YOLO(model_path)
                    print("[SUCCESS] License Plate Detector ready.")
                except Exception as e:
                    print(f"[WARN] Failed to load plate detector: {e}")
            else:
                print(f"[WARN] License plate model not found at {model_path}")
        return plate_model

# Stream Generators per Camera
class CameraStreamWorker:
    def __init__(self, camera_id: str, video_path: str, backend_url: str):
        self.camera_id = camera_id
        self.video_path = video_path
        self.backend_url = backend_url
        self.lock = threading.Lock()
        self.identity_pipeline = IdentityPipeline(use_gpu=False)
        self.tracker = VehicleTracker(camera_id=camera_id, identity_pipeline=self.identity_pipeline)
        self.backend_client = BackendClient(base_url=backend_url)
        self.current_jpeg: Optional[bytes] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False

    def _run_loop(self):
        print(f"[STREAM-WORKER] Starting video processing for {self.camera_id} ({self.video_path})...")
        p_detector = get_plate_model()
        latest_plates = []

        while self.is_running:
            stream = VideoStream(source=self.video_path, loop=True)
            try:
                for frame, frame_num, timestamp_sec in stream.read_frames():
                    if not self.is_running:
                        break

                    # 1. Update Tracker
                    active_tracks = self.tracker.update(frame, frame_num=frame_num, timestamp_sec=timestamp_sec)

                    # 2. Dispatch events to Backend
                    events = self.tracker.track_manager.get_dispatchable_events()
                    for ev in events:
                        self.backend_client.send_detection_event(ev)

                    # 3. Detect License Plate Boxes if Plate Model available
                    if p_detector is not None and frame_num % 3 == 0:
                        try:
                            plate_res = p_detector(frame, conf=0.22, verbose=False)[0]
                            if plate_res.boxes is not None and len(plate_res.boxes) > 0:
                                new_plates = []
                                for pbox in plate_res.boxes:
                                    px1, py1, px2, py2 = [int(v) for v in pbox.xyxy[0].tolist()]
                                    pconf = float(pbox.conf[0])
                                    new_plates.append((px1, py1, px2, py2, pconf))
                                latest_plates = new_plates
                        except Exception:
                            pass

                    # Draw red plate boxes
                    for px1, py1, px2, py2, pconf in latest_plates:
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        p_label = f"PLATE {int(pconf * 100)}%"
                        cv2.putText(
                            frame,
                            p_label,
                            (px1, max(py1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    # 4. Draw Vehicle Bounding Boxes & Trajectories
                    for track in active_tracks:
                        x1, y1, x2, y2 = track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2

                        if track.state == TrackState.ACTIVE:
                            box_color = (0, 255, 128)  # Neon Green
                            state_tag = f"{int(track.latest_confidence * 100)}%"
                        else:
                            box_color = (0, 220, 255)  # Cyan
                            state_tag = f"LOST"

                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        # Label Tag
                        if track.plate_number and not track.plate_number.startswith("TRACK_") and not track.plate_number.startswith("CAM_") and not track.plate_number.startswith("NO_PLATE") and not track.plate_number.startswith("UNREADABLE"):
                            label = f"{track.plate_number} [{int(track.plate_confidence * 100)}%]"
                        else:
                            tag_id = track.local_track_id.split("_")[-1]
                            label = f"{track.vehicle_type.upper()} #{tag_id} ({state_tag})"

                        # Draw Label Banner
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + lw + 8, y1), box_color, -1)
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 4, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                        # Draw motion trail
                        trail = track.trajectory
                        for i in range(1, len(trail)):
                            pt1 = (trail[i - 1][0], trail[i - 1][1])
                            pt2 = (trail[i][0], trail[i][1])
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

                    # 5. Encode frame to JPEG
                    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        with self.lock:
                            self.current_jpeg = buffer.tobytes()

                    # Regulate frame rate (~25 FPS)
                    time.sleep(0.038)
            except Exception as e:
                print(f"[STREAM-WORKER] Loop error for {self.camera_id}: {e}")
                time.sleep(1.0)
            finally:
                stream.release()

        self.backend_client.close()
        print(f"[STREAM-WORKER] Worker stopped for {self.camera_id}")

active_workers: Dict[str, CameraStreamWorker] = {}
workers_lock = threading.Lock()

def resolve_video_file(camera_id: str, requested_path: Optional[str]) -> str:
    possible_paths = []
    if requested_path:
        possible_paths.append(requested_path)
        base_name = os.path.basename(requested_path)
        possible_paths.append(os.path.join(BASE_DIR, "data", "videos", base_name))
        possible_paths.append(os.path.join(BASE_DIR, "..", "TraffixAI-F", "public", "videos", base_name))

    # Camera specific defaults
    if camera_id in ["CAM_002", "CAM_004"]:
        possible_paths.extend([
            os.path.join(BASE_DIR, "data", "videos", "junction_traffic.mp4"),
            os.path.join(BASE_DIR, "..", "TraffixAI-F", "public", "videos", "junction_traffic.mp4"),
            os.path.join(BASE_DIR, "data", "videos", "gettyimages-465302231-640_adpp.mp4"),
        ])
    else:
        possible_paths.extend([
            os.path.join(BASE_DIR, "data", "videos", "sample_traffic.mp4"),
            os.path.join(BASE_DIR, "..", "TraffixAI-F", "public", "videos", "sample_traffic.mp4"),
            os.path.join(BASE_DIR, "data", "videos", "gettyimages-1191315794-640_adpp.mp4"),
            os.path.join(BASE_DIR, "data", "videos", "215258_medium.mp4"),
        ])

    for p in possible_paths:
        if p and os.path.exists(p):
            return os.path.abspath(p)

    # Absolute fallback
    fallback_dir = os.path.join(BASE_DIR, "data", "videos")
    files = [os.path.join(fallback_dir, f) for f in os.listdir(fallback_dir) if f.endswith(".mp4")]
    if files:
        return files[0]

    return requested_path or ""

def get_or_create_worker(camera_id: str, video_path: str, backend_url: str) -> CameraStreamWorker:
    with workers_lock:
        if camera_id not in active_workers or active_workers[camera_id].video_path != video_path:
            if camera_id in active_workers:
                active_workers[camera_id].stop()
            worker = CameraStreamWorker(camera_id, video_path, backend_url)
            worker.start()
            active_workers[camera_id] = worker
        return active_workers[camera_id]

@app.get("/status")
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "traffix-ai-stream-daemon", "active_cameras": list(active_workers.keys())}

def frame_generator(worker: CameraStreamWorker) -> Generator[bytes, None, None]:
    while worker.is_running:
        with worker.lock:
            jpeg = worker.current_jpeg
        if jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.035)

@app.get("/api/v1/stream/{camera_id}")
def stream_camera(
    camera_id: str,
    video_path: Optional[str] = Query(None),
    backend_url: Optional[str] = Query("http://localhost:8000")
):
    resolved_video = resolve_video_file(camera_id, video_path)
    worker = get_or_create_worker(camera_id, resolved_video, backend_url)
    return StreamingResponse(
        frame_generator(worker),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    print("=" * 65)
    print("      TRAFFIX AI — LIVE STREAM DAEMON (PORT 8002)      ")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
