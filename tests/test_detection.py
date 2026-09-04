import cv2
import time
import os
from ultralytics import YOLO

# COCO Dataset class IDs for road vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def run_vehicle_detection(video_path: str, model_name: str = "yolov8n.pt", conf_threshold: float = 0.4):
    """
    Minimal test script to run YOLO vehicle detection on a video stream,
    filter only road vehicles, draw bounding boxes using OpenCV,
    and calculate real-time inference FPS.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    # 1. Load the lightweight YOLO nano model (downloads ~6MB weights automatically on first run)
    print(f"[INFO] Loading YOLO model: {model_name}...")
    model = YOLO(model_name)

    # 2. Open the video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_native = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 55)
    print("      YOLO VEHICLE DETECTION PIPELINE ACTIVE      ")
    print("=" * 55)
    print(f"Video Source      : {video_path} ({width}x{height} @ {fps_native:.1f} FPS)")
    print(f"Target Classes    : {list(VEHICLE_CLASSES.values())}")
    print(f"Confidence Filter : >= {conf_threshold * 100}%")
    print("=" * 55)
    print("Press 'q' to stop.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[INFO] Video playback completed.")
            break

        frame_count += 1
        infer_start = time.time()

        # 3. Run YOLO inference on the current frame
        # verbose=False suppresses per-frame console spam
        results = model(frame, verbose=False)[0]

        infer_latency_ms = (time.time() - infer_start) * 1000

        detected_vehicles = 0

        # 4. Extract bounding boxes, confidence scores, and class IDs
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # Filter: Check if the detected object is a vehicle and meets confidence threshold
            if class_id in VEHICLE_CLASSES and confidence >= conf_threshold:
                detected_vehicles += 1
                vehicle_label = VEHICLE_CLASSES[class_id]

                # Coordinates: [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # 5. Draw bounding box with OpenCV (Green box for cars, Cyan for trucks/buses, etc.)
                box_color = (0, 255, 0) if class_id == 2 else (255, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw label background and text above the box
                label_text = f"{vehicle_label} {confidence:.2f}"
                cv2.putText(
                    frame,
                    label_text,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                    cv2.LINE_AA
                )

        # 6. Overall FPS and performance metrics overlay
        elapsed = time.time() - start_time
        total_fps = frame_count / elapsed if elapsed > 0 else 0

        hud_text = f"FPS: {total_fps:.1f} | Latency: {infer_latency_ms:.1f}ms | Vehicles: {detected_vehicles}"
        cv2.putText(frame, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 7. Render frame
        cv2.imshow("Traffix AI - YOLO Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Finished detection test. Average FPS: {total_fps:.1f}")

if __name__ == "__main__":
    from config.settings import settings
    run_vehicle_detection(settings.VIDEO_SOURCE)
