# Traffix AI — Perception & Identity Engine (AI Service)

> **Smart India Hackathon (SIH 2026) | Problem Statement: PS 26127**  
> *Sponsor: Bharat Electronics Limited (BEL)*  
> **Centralized City-Wide Multi-Camera ANPR Trajectory Tracking System**

---

## 📌 Project Overview

**Traffix AI** processes multi-camera RTSP/video feeds across a city network. This repository houses the **AI Perception & Identity Service**, collaboratively owned by:

* **Person 1 (Perception & Motion)**: Video ingest, YOLOv8/v11 vehicle detection, BoT-SORT/ByteTrack multi-object tracking, track lifecycle management (`CAM_001_T_x`), and FPS optimization.
* **Person 2 (Identity Intelligence)**: Automatic Number Plate Recognition (ANPR), character preprocessing (CLAHE + Bilateral Filtering), deep learning OCR (EasyOCR), Indian license plate syntax validation & correction, and Deep Metric Appearance Re-ID Embeddings.

```
┌─────────────────────────────────────────────────────────┐
│              TRAFFIX AI — AI SERVICE                    │
│                                                         │
│  [ Camera / RTSP ]                                      │
│         │                                               │
│         ▼                                               │
│  [ YOLO Detection ] ──► [ BoT-SORT Tracking ]           │
│                                  │                      │
│                                  ▼                      │
│                  [ Best Vehicle Crop Extraction ]       │
│                                  │                      │
│         ┌────────────────────────┴───────────────────┐  │
│         ▼                                            ▼  │
│  [ ANPR & OCR Engine ]                      [ Vehicle Re-ID ]
│  • CLAHE + Bilateral Filter                 • MobileNetV3    │
│  • EasyOCR Text Extraction                  • 1024-D Vector  │
│  • Indian Syntax Correction                 • L2 Normalization
│         │                                            │  │
│         └────────────────────────┬───────────────────┘  │
│                                  ▼                      │
│                  [ Consolidated DetectionEvent ]        │
└──────────────────────────────────┬──────────────────────┘
                                   │ POST /api/v1/events/detection
                                   ▼
┌─────────────────────────────────────────────────────────┐
│        BACKEND SERVICE (Database & Trajectory)          │
│   • PostgreSQL / PostGIS / Vector Storage               │
│   • HMM Trajectory Reconstruction & Multi-Camera Graph  │
└─────────────────────────────────────────────────────────┘
                                   │ REST + WebSockets
                                   ▼
┌─────────────────────────────────────────────────────────┐
│        FRONTEND (City Map & Analytics Dashboard)        │
│   • React + TypeScript + MapLibre GIS                   │
└─────────────────────────────────────────────────────────┘
```

---

## 💻 Prerequisites

Ensure you have the following installed on your machine:

1. **Python 3.10, 3.11, or 3.12** (Python 3.12 recommended).
   * Verify via terminal: `python --version` (or `python3 --version`).
2. **Git**:
   * Verify via terminal: `git --version`.
3. **Hardware**:
   * Runs efficiently on **standard CPU** (tested and optimized at ~9 ms Re-ID inference and ~25 FPS tracking on CPU).
   * GPU (NVIDIA CUDA) is automatically utilized if available.

---

## 🚀 Setup & Installation Guide

Choose your Operating System below:

### 🐧 Option A: Linux (Ubuntu / Fedora / Debian)

#### 1. Open Terminal and Navigate to the Repository
```bash
cd "Traffix Ai"
```

#### 2. Create a Python Virtual Environment
```bash
python3 -m venv venv
```

#### 3. Activate the Virtual Environment
```bash
source venv/bin/activate
```
*(Your terminal prompt should now show `(venv)` at the beginning).*

#### 4. Upgrade Pip and Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 🪟 Option B: Windows (PowerShell / Command Prompt)

#### 1. Open PowerShell or Terminal
Open PowerShell or Command Prompt and navigate into the repository:
```powershell
cd "Traffix Ai"
```

#### 2. Create a Python Virtual Environment
```powershell
python -m venv venv
```

#### 3. Activate the Virtual Environment

* **In PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  > ⚠️ **PowerShell Execution Policy Note**:  
  > If Windows blocks script execution (`running scripts is disabled on this system`), run this command once to allow local scripts for your session:
  > ```powershell
  > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  > .\venv\Scripts\Activate.ps1
  > ```

* **In Command Prompt (cmd.exe)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

*(Your terminal prompt should now show `(venv)`).*

#### 4. Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration (`.env`)

The system loads runtime configurations from a `.env` file.

1. **Copy the example configuration**:
   * **Linux**:
     ```bash
     cp .env.example .env
     ```
   * **Windows (PowerShell)**:
     ```powershell
     Copy-Item .env.example .env
     ```

2. **Configuration Options in `.env`**:
   ```ini
   # Camera Configuration
   CAMERA_ID=CAM_001
   CAMERA_NAME=Junction A
   CAMERA_LATITUDE=22.5726
   CAMERA_LONGITUDE=88.3639
   VIDEO_SOURCE=data/videos/sample_traffic.mp4   # File path, RTSP url, or '0' for Webcam

   # YOLO & Tracker Parameters
   YOLO_MODEL=yolov8n.pt                        # Pretrained nano weights (~6 MB)
   CONF_THRESHOLD=0.20
   TRACKER_CONFIG=config/tracker.yaml
   MIN_HITS_TO_CONFIRM=4                        # Frames required to confirm a real vehicle
   MIN_BOX_AREA=625                             # Filters out small background noise
   MAX_LOST_FRAMES=20                           # Frames to coast through temporary occlusion

   # Backend API Configuration (Person 3)
   BACKEND_URL=http://localhost:8000
   HEARTBEAT_INTERVAL_SEC=15
   ```

---

## 🧪 Verification & Unit Tests

Run these targeted test scripts to verify each component independently before launching the full pipeline:

### 1. Validate Shared API Contracts
Ensures detection events and camera heartbeat schemas match the backend contract:
```bash
python tests/test_contract.py
```

### 2. Test Camera & Video Ingestion
Tests video stream opening, frame decoding, and FPS measurement:
```bash
python tests/test_camera.py
```

### 3. Test YOLO Vehicle Detection
Tests vehicle detection (car, motorcycle, bus, truck) on video frames:
```bash
python tests/test_detection.py
```

### 4. Test Multi-Object Tracking & Lifecycle
Tests BoT-SORT tracking, trajectory generation, and state transitions (`NEW` $\to$ `ACTIVE` $\to$ `LOST` $\to$ `ENDED`):
```bash
python tests/test_tracking.py
```

### 5. Test ANPR & Indian License Plate Syntax Correction
Tests CLAHE + Bilateral preprocessing, EasyOCR reading, and positional syntax rectification (e.g. correcting optical OCR errors like `Z` $\to$ `2` based on Indian RTO registration rules):
```bash
python tests/test_anpr.py
```

---

## 🎬 Running the Master AI Service

To run the complete real-time pipeline:

```bash
python main.py
```

### What Happens When You Run:
1. **Camera Ingestion**: Reads the configured video source in a continuous loop.
2. **Detection & Tracking**: YOLO detects vehicles; BoT-SORT maintains persistent local track IDs (`CAM_001_T_x`).
3. **Consolidated Events**: When a vehicle track is confirmed `ACTIVE`, the best high-resolution crop is evaluated.
4. **Backend Dispatch**: Emits a structured `DetectionEvent` to the backend. If the backend is offline, the event is safely printed to the console without crashing.
5. **Camera Heartbeat**: Automatically pings the backend every 15 seconds with camera status, live FPS, and latency.
6. **Visual Overlay Window**: Displays active tracks (Green), coasting tracks (Cyan), motion trajectory trails, and a real-time HUD.

Press **`q`** in the video display window to stop the pipeline gracefully.

---

## 📂 Project Architecture

```
Traffix Ai/
├── camera/                  # Camera & video ingestion stream wrapper
│   └── stream.py
├── client/                  # HTTP client for Backend API & Heartbeat
│   └── backend_client.py
├── config/                  # Pydantic environment settings & tracker YAML
│   ├── settings.py
│   └── tracker.yaml
├── data/
│   └── videos/              # Sample traffic videos for local testing
├── detection/               # YOLO vehicle detection wrapper
│   └── yolo_detector.py
├── tracking/                # BoT-SORT tracker & Track Lifecycle State Machine
│   ├── tracker.py
│   └── track_manager.py
├── anpr/                    # [Person 2] License plate detection, OCR & Syntax engine
│   ├── preprocessor.py
│   ├── plate_parser.py
│   └── ocr_engine.py
├── reid/                    # [Person 2] Deep metric visual appearance embeddings
│   ├── feature_extractor.py
│   └── distance.py
├── schemas/                 # Pydantic data contracts (Single Source of Truth)
│   ├── observation.py       # Contract: Person 1 -> Person 2
│   ├── event.py             # Contract: AI Service -> Backend
│   ├── heartbeat.py         # Contract: Camera Heartbeat -> Backend
│   └── track.py             # Internal track lifecycle model
├── tests/                   # Modular test suite for CI and local verification
│   ├── test_contract.py
│   ├── test_camera.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   └── test_anpr.py
├── requirements.txt         # Pinned Python package dependencies
└── main.py                  # Master entry point orchestrating the AI service
```

---

## 🛠️ Troubleshooting & FAQ

### 1. Windows: `Activate.ps1 cannot be loaded because running scripts is disabled`
* **Cause**: Windows PowerShell security policy restricts script execution by default.
* **Fix**: Open PowerShell and run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  Then re-run `.\venv\Scripts\Activate.ps1`.

### 2. Windows: Missing Microsoft Visual C++ Redistributable
* **Cause**: PyTorch or OpenCV can throw `DLL load failed` if the Visual C++ runtime is missing.
* **Fix**: Download and install the free official **[Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)**, then restart your terminal.

### 3. Linux Headless / Docker: `ImportError: libGL.so.1: cannot open shared object file`
* **Cause**: OpenCV requires OpenGL libraries that are omitted in minimal server environments.
* **Fix**: Run:
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
  ```

### 4. `[CONSOLIDATED EVENT] ... -> Generated (Backend Offline)`
* **Explanation**: This is normal during standalone AI testing. The AI service checks if your teammate's backend (`BACKEND_URL`) is reachable. If offline, it logs the event locally and continues streaming without interruption.

### 5. CPU vs GPU Performance
* The pipeline automatically detects if an NVIDIA GPU with CUDA is available. If running on CPU, the default models (`yolov8n.pt` and `mobilenet_v3_small`) are specifically chosen for sub-15ms inference on standard multi-core CPUs.
