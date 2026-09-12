import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
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
anpr_engine_instance = None
anpr_engine_lock = threading.Lock()

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

def get_anpr_engine():
    global anpr_engine_instance
    with anpr_engine_lock:
        if anpr_engine_instance is None:
            from anpr.ocr_engine import ANPREngine
            try:
                anpr_engine_instance = ANPREngine(gpu=False)
            except Exception as e:
                print(f"[WARN] Failed to load ANPREngine: {e}")
        return anpr_engine_instance

# Stream Generators per Camera
class CameraStreamWorker:
    def __init__(self, camera_id: str, video_path: str, backend_url: str):
        self.camera_id = camera_id
        self.video_path = video_path
        self.backend_url = backend_url
        self.lock = threading.Lock()
        self.tracker = VehicleTracker(camera_id=camera_id, identity_pipeline=None)
        self.backend_client = BackendClient(base_url=backend_url)
        self.current_jpeg: Optional[bytes] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.ocr_queue = queue.Queue(maxsize=8)
        self.ocr_thread: Optional[threading.Thread] = None
        self.event_queue = queue.Queue(maxsize=50)
        self.event_thread: Optional[threading.Thread] = None
        self.track_ocr_votes: Dict[str, Dict[str, int]] = {}

    def _event_worker_loop(self):
        while self.is_running:
            try:
                ev = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.backend_client.send_detection_event(ev)
            except Exception:
                pass
            finally:
                self.event_queue.task_done()

    def _ocr_worker_loop(self):
        engine = get_anpr_engine()
        while self.is_running:
            try:
                task = self.ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            crop, track_ref = task
            try:
                if engine is not None:
                    plate_text, ocr_conf = engine.read_plate(crop)
                else:
                    plate_text, ocr_conf = None, None

                if plate_text and len(plate_text) >= 5 and track_ref:
                    track_id = track_ref.local_track_id
                    if track_id not in self.track_ocr_votes:
                        self.track_ocr_votes[track_id] = {}
                    self.track_ocr_votes[track_id][plate_text] = self.track_ocr_votes[track_id].get(plate_text, 0) + 1
                    vote_count = self.track_ocr_votes[track_id][plate_text]

                    # Accept if validated high confidence (>= 0.85) or confirmed across >= 2 readings
                    should_accept = (ocr_conf and ocr_conf >= 0.85) or vote_count >= 2

                    if should_accept:
                        track_ref.plate_number = plate_text
                        track_ref.plate_confidence = ocr_conf or 0.90
                        tag = track_id.split('_')[-1]
                        if hasattr(self.tracker, 'track_manager') and track_id in self.tracker.track_manager.tracks:
                            rec = self.tracker.track_manager.tracks[track_id]
                            rec.plate_number = plate_text
                            rec.plate_confidence = track_ref.plate_confidence
                        print(f"\033[1;32m[AI-ANPR] 🎯 RECOGNIZED PLATE: [{plate_text}] (Conf: {int((ocr_conf or 0.90)*100)}%) on {track_ref.vehicle_type.upper()} #{tag} @ {self.camera_id}\033[0m", flush=True)

                        # Dispatch real-time update with recognized plate
                        from schemas.event import DetectionEvent
                        now_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                        upd_ev = DetectionEvent(
                            camera_id=self.camera_id,
                            observed_at=now_iso,
                            local_track_id=track_ref.local_track_id,
                            vehicle_type=track_ref.vehicle_type,
                            vehicle_confidence=track_ref.latest_confidence,
                            bounding_box=track_ref.bbox,
                            plate_number=plate_text,
                            plate_confidence=track_ref.plate_confidence,
                            vehicle_embedding=[],
                            first_seen=track_ref.first_seen,
                            last_seen=now_iso,
                            status="active"
                        )
                        try:
                            self.event_queue.put_nowait(upd_ev)
                        except queue.Full:
                            pass
            except Exception:
                pass
            finally:
                self.ocr_queue.task_done()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.event_thread = threading.Thread(target=self._event_worker_loop, daemon=True)
            self.event_thread.start()
            self.ocr_thread = threading.Thread(target=self._ocr_worker_loop, daemon=True)
            self.ocr_thread.start()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False

    def _run_loop(self):
        print(f"\033[1;36m[AI-VISION] 🎥 Processing live stream for {self.camera_id} from {self.video_path}\033[0m", flush=True)
        p_detector = get_plate_model()
        latest_plates = []

        while self.is_running:
            stream = VideoStream(source=self.video_path, loop=True)
            try:
                for frame, frame_num, timestamp_sec in stream.read_frames():
                    if not self.is_running:
                        break

                    # 1. Update Tracker (Fast YOLO Tracking without blocking OCR)
                    active_tracks = self.tracker.update(frame, frame_num=frame_num, timestamp_sec=timestamp_sec)

                    if frame_num % 35 == 0 and len(active_tracks) > 0:
                        plates_list = [t.plate_number for t in active_tracks if t.plate_number and not t.plate_number.startswith("TRACK_") and not t.plate_number.startswith("NO_PLATE")]
                        plates_str = f" | 🏷️ Plates: {', '.join(plates_list)}" if plates_list else ""
                        print(f"\033[1;34m[AI-VISION]\033[0m 🚗 [{self.camera_id}] Tracking {len(active_tracks)} vehicles{plates_str} | Frame {frame_num}", flush=True)

                    # 2. Dispatch events to Backend via non-blocking queue
                    events = self.tracker.track_manager.get_dispatchable_events()
                    for ev in events:
                        try:
                            self.event_queue.put_nowait(ev)
                        except queue.Full:
                            pass

                    # 3. Detect License Plate Boxes with YOLO (every 4 frames for high stream FPS)
                    if p_detector is not None and frame_num % 4 == 0:
                        try:
                            plate_res = p_detector(frame, conf=0.18, verbose=False)[0]
                            if plate_res.boxes is not None and len(plate_res.boxes) > 0:
                                new_plates = []
                                for pbox in plate_res.boxes:
                                    px1, py1, px2, py2 = [int(v) for v in pbox.xyxy[0].tolist()]
                                    pconf = float(pbox.conf[0])

                                    # Find matching track
                                    pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                                    matched_track = None
                                    for track in active_tracks:
                                        if (track.bbox.x1 - 25 <= pcx <= track.bbox.x2 + 25 and
                                            track.bbox.y1 - 25 <= pcy <= track.bbox.y2 + 25):
                                            matched_track = track
                                            break

                                    # Queue OCR if track has no valid readable plate yet and crop is large enough
                                    has_clean_plate = bool(
                                        matched_track
                                        and matched_track.plate_number
                                        and not matched_track.plate_number.startswith("TRACK_")
                                        and not matched_track.plate_number.startswith("CAM_")
                                        and not matched_track.plate_number.startswith("UNREADABLE")
                                        and not matched_track.plate_number.startswith("NO_PLATE")
                                    )
                                    if matched_track and not has_clean_plate:
                                        if (py2 - py1) >= 12 and (px2 - px1) >= 25 and not self.ocr_queue.full():
                                            pad = 6
                                            crop = frame[max(0, py1 - pad):min(frame.shape[0], py2 + pad), max(0, px1 - pad):min(frame.shape[1], px2 + pad)].copy()
                                            if crop.size > 0:
                                                try:
                                                    self.ocr_queue.put_nowait((crop, matched_track))
                                                except queue.Full:
                                                    pass

                                    new_plates.append((px1, py1, px2, py2, pconf, matched_track))
                                latest_plates = new_plates
                        except Exception as e:
                            pass

                    # Draw sharp red plate boxes with recognized registration
                    for plate_info in latest_plates:
                        if len(plate_info) == 6:
                            px1, py1, px2, py2, pconf, matched_track = plate_info
                        else:
                            px1, py1, px2, py2, pconf = plate_info[:5]
                            matched_track = None

                        p_text = None
                        if matched_track and matched_track.plate_number and not matched_track.plate_number.startswith("TRACK_") and not matched_track.plate_number.startswith("NO_PLATE") and not matched_track.plate_number.startswith("UNREADABLE"):
                            p_text = matched_track.plate_number

                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        p_label = f"PLATE: {p_text}" if p_text else f"PLATE {int(pconf * 100)}%"
                        (plw, plh), _ = cv2.getTextSize(p_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                        cv2.rectangle(frame, (px1, max(0, py1 - 16)), (px1 + plw + 4, py1), (0, 0, 255), -1)
                        cv2.putText(
                            frame,
                            p_label,
                            (px1 + 2, max(12, py1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (255, 255, 255),
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

                    # Regulate frame rate (~30 FPS playback)
                    time.sleep(0.026)
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
    frontend_videos_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "TraffixAI-F", "public", "videos"))
    data_videos_dir = os.path.abspath(os.path.join(BASE_DIR, "data", "videos"))

    if requested_path:
        base_name = os.path.basename(requested_path)
        possible_paths.extend([
            requested_path,
            os.path.join(frontend_videos_dir, base_name),
            os.path.join(data_videos_dir, base_name),
            os.path.join(frontend_videos_dir, requested_path.lstrip("/")),
            os.path.join(data_videos_dir, requested_path.lstrip("/")),
        ])

    # Camera specific defaults
    if camera_id in ["CAM_002", "CAM_004"]:
        possible_paths.extend([
            os.path.join(frontend_videos_dir, "junction_traffic.mp4"),
            os.path.join(frontend_videos_dir, "gettyimages-465302231-640_adpp.mp4"),
            os.path.join(frontend_videos_dir, "sample_traffic.mp4"),
            os.path.join(frontend_videos_dir, "gettyimages-1191315794-640_adpp.mp4"),
            os.path.join(data_videos_dir, "junction_traffic.mp4"),
            os.path.join(data_videos_dir, "gettyimages-465302231-640_adpp.mp4"),
        ])
    else:
        possible_paths.extend([
            os.path.join(frontend_videos_dir, "sample_traffic.mp4"),
            os.path.join(frontend_videos_dir, "gettyimages-1191315794-640_adpp.mp4"),
            os.path.join(frontend_videos_dir, "junction_traffic.mp4"),
            os.path.join(frontend_videos_dir, "gettyimages-465302231-640_adpp.mp4"),
            os.path.join(data_videos_dir, "sample_traffic.mp4"),
            os.path.join(data_videos_dir, "gettyimages-1191315794-640_adpp.mp4"),
            os.path.join(data_videos_dir, "215258_medium.mp4"),
        ])

    for p in possible_paths:
        if p and os.path.exists(p) and os.path.isfile(p):
            return os.path.abspath(p)

    # Absolute fallback: scan frontend then data
    for scan_dir in [frontend_videos_dir, data_videos_dir]:
        if os.path.exists(scan_dir):
            files = [os.path.join(scan_dir, f) for f in os.listdir(scan_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))]
            if files:
                return os.path.abspath(files[0])

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

@app.get("/api/v1/videos")
def list_videos():
    frontend_videos_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "TraffixAI-F", "public", "videos"))
    video_map = {}
    valid_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    if os.path.exists(frontend_videos_dir):
        for f in sorted(os.listdir(frontend_videos_dir)):
            if f.lower().endswith(valid_exts) and f not in video_map:
                fp = os.path.join(frontend_videos_dir, f)
                st = os.stat(fp)
                video_map[f] = {
                    "id": f,
                    "filename": f,
                    "name": f.replace("-", " ").replace("_", " ").rsplit(".", 1)[0],
                    "path": f"/videos/{f}",
                    "sizeBytes": st.st_size,
                    "sizeFormatted": f"{st.st_size / (1024 * 1024):.1f} MB",
                    "diskPath": fp
                }
    return {"success": True, "data": {"videos": list(video_map.values()), "total": len(video_map)}}

def frame_generator(worker: CameraStreamWorker) -> Generator[bytes, None, None]:
    for _ in range(50):
        if worker.current_jpeg is not None or not worker.is_running:
            break
        time.sleep(0.04)

    while worker.is_running:
        with worker.lock:
            jpeg = worker.current_jpeg
        if jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
                jpeg +
                b"\r\n"
            )
        time.sleep(0.035)

from pydantic import BaseModel

class DetectRequest(BaseModel):
    camera_id: str
    video_source: Optional[str] = None
    backend_url: Optional[str] = "http://localhost:8000"
    max_frames: Optional[int] = None

@app.post("/api/v1/detect")
def trigger_detection(req: DetectRequest):
    resolved_video = resolve_video_file(req.camera_id, req.video_source)
    worker = get_or_create_worker(req.camera_id, resolved_video, req.backend_url or "http://localhost:8000")
    return {
        "success": True,
        "message": f"Continuous AI detection active for {req.camera_id}",
        "camera_id": req.camera_id,
        "video_path": resolved_video,
    }

@app.get("/api/v1/snapshot/{camera_id}")
def snapshot_camera(
    camera_id: str,
    video_path: Optional[str] = Query(None),
    backend_url: Optional[str] = Query("http://localhost:8000")
):
    resolved_video = resolve_video_file(camera_id, video_path)
    worker = get_or_create_worker(camera_id, resolved_video, backend_url)
    for _ in range(40):
        with worker.lock:
            jpeg = worker.current_jpeg
        if jpeg is not None:
            return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        time.sleep(0.05)
    return Response(content=b"", status_code=503)

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
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close"
        }
    )

@app.post("/api/v1/stream/stop/{camera_id}")
def stop_stream_camera(camera_id: str):
    with workers_lock:
        if camera_id in active_workers:
            worker = active_workers.pop(camera_id)
            worker.stop()
            print(f"[STREAM-SERVER] Stopped stream for {camera_id}")
            return {"success": True, "message": f"Stream worker for {camera_id} stopped."}
        return {"success": True, "message": f"No active stream for {camera_id}."}

if __name__ == "__main__":
    print("=" * 65)
    print("      TRAFFIX AI — LIVE STREAM DAEMON (PORT 8002)      ")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
