import cv2
import time
import os

def test_video_stream(video_path: str):
    """
    Test script to verify OpenCV video ingestion, frame decoding,
    metadata extraction, and real-time FPS calculation.
    """
    # 1. Check if the video file actually exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    # 2. Open the video source using cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)

    # 3. Verify that the video was opened successfully
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video stream: {video_path}")
        return

    # 4. Extract video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print("=" * 50)
    print("       VIDEO STREAM METADATA EXTRACTED       ")
    print("=" * 50)
    print(f"Source Path     : {video_path}")
    print(f"Resolution      : {width} x {height}")
    print(f"Native Video FPS: {fps:.2f}")
    print(f"Total Frames    : {total_frames}")
    print(f"Duration        : {duration_sec:.2f} seconds")
    print("=" * 50)
    print("Starting playback window. Press 'q' to stop.")

    frame_count = 0
    start_time = time.time()

    # 5. Read frames in a loop
    while True:
        # ret: boolean (True if frame read successfully)
        # frame: 3D NumPy array (Height x Width x 3 BGR channels)
        ret, frame = cap.read()

        if not ret:
            print("\n[INFO] End of video stream reached.")
            break

        frame_count += 1

        # 6. Calculate real-time processing FPS
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        current_time_sec = frame_count / fps if fps > 0 else 0

        # 7. Draw an informative overlay on top of the frame
        overlay_text = f"Frame: {frame_count}/{total_frames} | Time: {current_time_sec:.2f}s | FPS: {current_fps:.1f}"
        cv2.putText(
            img=frame,
            text=overlay_text,
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0),  # Green text in BGR format
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # 8. Display the frame in an OpenCV window
        cv2.imshow("Traffix AI - Video Ingestion Test", frame)

        # 9. Wait for key press (1 ms delay). If 'q' is pressed, break out of loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] User pressed 'q'. Exiting...")
            break

    # 10. Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Processed {frame_count} frames in {elapsed_time:.2f}s at average {current_fps:.1f} FPS.")

if __name__ == "__main__":
    sample_video = "data/videos/sample_traffic.mp4"
    test_video_stream(sample_video)
