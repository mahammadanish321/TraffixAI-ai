import cv2
import time
import os
from collections import defaultdict
from ultralytics import YOLO

# COCO road vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Production tracking thresholds
MIN_HITS_TO_CONFIRM = 3      # Confirms vehicle after 3 consecutive detections
MIN_BOX_AREA = 25 * 25       # Ignore tiny boxes smaller than 25x25 pixels (noise in distant background)


def run_vehicle_tracking(
    video_path: str,
    camera_id: str = "CAM_001",
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.35
):
    """
    Production-ready vehicle tracking test with:
    - ByteTrack persistent tracking
    - Minimum hit confirmation (filters flickering noise)
    - Minimum bounding box size filtering
    - Track lifecycle stats and trajectory trails
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    print(f"[INFO] Initializing YOLO + ByteTrack on {video_path}...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    # Track lifecycle state:
    # track_hits: track_id -> number of times seen
    track_hits = defaultdict(int)
    # track_history: track_id -> list of (center_x, center_y) trail points
    track_history = defaultdict(list)
    # confirmed_vehicles: set of track_ids that met confirmation threshold
    confirmed_vehicles = set()

    print("=" * 65)
    print(f"       ROBUST VEHICLE TRACKING PIPELINE ACTIVE ({camera_id})       ")
    print("=" * 65)
    print(f"Min Hits to Confirm: {MIN_HITS_TO_CONFIRM} frames")
    print(f"Min Box Area       : {MIN_BOX_AREA} px")
    print(f"Confidence Filter  : >= {conf_threshold * 100}%")
    print("=" * 65)
    print("Press 'q' to stop.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[INFO] Video stream ended.")
            break

        frame_count += 1

        # Run YOLO + ByteTrack
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )[0]

        active_confirmed_in_frame = 0

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()
            class_ids = results.boxes.cls.int().cpu().tolist()
            confidences = results.boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                box_area = width * height

                # Filter 1: Ignore tiny distant bounding boxes
                if box_area < MIN_BOX_AREA:
                    continue

                # Increment track hits count
                track_hits[track_id] += 1

                # Filter 2: Check if track has been observed enough times to confirm
                if track_hits[track_id] >= MIN_HITS_TO_CONFIRM:
                    confirmed_vehicles.add(track_id)
                    active_confirmed_in_frame += 1

                    local_track_id = f"{camera_id}_T_{track_id}"
                    vehicle_type = VEHICLE_CLASSES.get(cls_id, "vehicle")

                    # Center bottom road contact point
                    center_x = int((x1 + x2) / 2)
                    center_y = int(y2)
                    track_history[track_id].append((center_x, center_y))
                    if len(track_history[track_id]) > 40:
                        track_history[track_id].pop(0)

                    # Draw Bounding Box & Label
                    box_color = (0, 255, 128) if cls_id == 2 else (255, 128, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    label = f"{local_track_id} | {vehicle_type} {conf:.2f}"
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

                    # Draw Trajectory Trail
                    points = track_history[track_id]
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)

        # FPS & HUD overlay
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        hud_text = f"FPS: {fps:.1f} | Active Vehicles: {active_confirmed_in_frame} | Confirmed Count: {len(confirmed_vehicles)}"
        cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Traffix AI - Robust Vehicle Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("=" * 65)
    print("              TRACK LIFECYCLE SUMMARY              ")
    print("=" * 65)
    print(f"Total Video Frames            : {frame_count}")
    print(f"Total Raw Tracks Detected     : {len(track_hits)}")
    print(f"Total Confirmed Vehicles      : {len(confirmed_vehicles)}")
    print("-" * 65)
    print("Track ID | Frame Hits | Status")
    print("-" * 65)
    for tid, hits in sorted(track_hits.items(), key=lambda item: item[0]):
        status = "✅ Confirmed Vehicle" if tid in confirmed_vehicles else "❌ Rejected (Noise/Too Short)"
        print(f"T_{tid:<6} | {hits:<10} | {status}")
    print("=" * 65)

if __name__ == "__main__":
    sample_video = "data/videos/sample_traffic.mp4"
    run_vehicle_tracking(sample_video, camera_id="CAM_001")
