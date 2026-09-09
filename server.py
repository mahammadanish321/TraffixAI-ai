"""
Traffix AI — Standby / Sleep-Mode Daemon Server
Listens on port 8002.
Default State: Sleep Mode (0% CPU / 0% GPU, listening on HTTP port).
On-Demand: Wakes up when POST /api/v1/detect is called, executes YOLOv8 + BoT-SORT + EasyOCR + Re-ID,
dispatches confirmed sightings to Backend (port 8000), and automatically returns to Sleep Mode.
"""

import sys
import os
import json
import time
import threading
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.settings import settings
from main import run_ai_service

# Global State
SERVER_LOCK = threading.Lock()
AI_STATE = {
    "state": "sleep",
    "current_camera": None,
    "last_run": None,
    "total_runs": 0,
}


class TraffixDaemonHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/api/v1/health"):
            self._send_json(200, {
                "status": "ok",
                "service": "Traffix_Ai Standby Daemon",
                "version": "1.0.0",
                "state": AI_STATE["state"]
            })
        elif self.path in ("/status", "/api/v1/status"):
            with SERVER_LOCK:
                self._send_json(200, {
                    "success": True,
                    "state": AI_STATE["state"],
                    "current_camera": AI_STATE["current_camera"],
                    "last_run": AI_STATE["last_run"],
                    "total_runs": AI_STATE["total_runs"]
                })
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path in ("/api/v1/detect", "/detect"):
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                payload = json.loads(raw_body)
            except Exception:
                payload = {}

            camera_id = payload.get("camera_id", "CAM_001")
            video_source = payload.get("video_source") or settings.VIDEO_SOURCE
            backend_url = payload.get("backend_url") or settings.BACKEND_URL
            max_frames = int(payload.get("max_frames", 60))

            # Resolve video file path
            resolved_video = video_source
            if not os.path.isabs(resolved_video):
                local_candidate = os.path.join(BASE_DIR, resolved_video)
                if os.path.exists(local_candidate):
                    resolved_video = local_candidate
                else:
                    data_candidate = os.path.join(BASE_DIR, "data", "videos", os.path.basename(resolved_video))
                    if os.path.exists(data_candidate):
                        resolved_video = data_candidate

            if not os.path.exists(resolved_video):
                default_video = os.path.join(BASE_DIR, "data", "videos", "215258_medium.mp4")
                if os.path.exists(default_video):
                    resolved_video = default_video

            print("\n" + "=" * 60)
            print(f" [AI STANDBY DAEMON] WAKING UP for Camera: {camera_id}")
            print(f" Source Video : {resolved_video}")
            print(f" Target Backend: {backend_url}")
            print("=" * 60)

            with SERVER_LOCK:
                if AI_STATE["state"] == "detecting":
                    self._send_json(409, {
                        "success": False,
                        "message": f"AI is already scanning camera {AI_STATE['current_camera']}. Please wait."
                    })
                    return

                AI_STATE["state"] = "detecting"
                AI_STATE["current_camera"] = camera_id

            try:
                events_count = run_ai_service(
                    camera_id=camera_id,
                    video_source=resolved_video,
                    backend_url=backend_url,
                    headless=True,
                    loop=False,
                    max_frames=max_frames
                )

                with SERVER_LOCK:
                    AI_STATE["state"] = "sleep"
                    AI_STATE["current_camera"] = None
                    AI_STATE["total_runs"] += 1
                    AI_STATE["last_run"] = {
                        "camera_id": camera_id,
                        "events_dispatched": events_count,
                        "timestamp": time.time()
                    }

                print("=" * 60)
                print(f" [AI STANDBY DAEMON] Returning to SLEEP MODE (0% CPU/GPU)")
                print("=" * 60 + "\n")

                self._send_json(200, {
                    "success": True,
                    "status": "completed",
                    "camera_id": camera_id,
                    "events_dispatched": events_count,
                    "state": "sleep"
                })

            except Exception as ex:
                with SERVER_LOCK:
                    AI_STATE["state"] = "sleep"
                    AI_STATE["current_camera"] = None

                print(f"[ERROR] Detection execution failed: {ex}")
                self._send_json(500, {
                    "success": False,
                    "error": str(ex),
                    "state": "sleep"
                })
        else:
            self._send_json(404, {"error": "Endpoint Not Found"})


def run_server(port: int = 8002):
    server_address = ("", port)
    httpd = HTTPServer(server_address, TraffixDaemonHandler)
    print("=" * 65)
    print(f"      TRAFFIX AI — STANDBY SLEEP-MODE DAEMON STARTED")
    print(f"      Listening on: http://localhost:{port}")
    print(f"      Current State: SLEEP (Idle, 0% CPU, 0% GPU)")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Daemon stopping...")
    finally:
        httpd.server_close()
        print("[INFO] Daemon shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffix AI Standby Daemon")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on (default: 8002)")
    args = parser.parse_args()
    run_server(args.port)
