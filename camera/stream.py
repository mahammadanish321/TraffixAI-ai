import cv2
import time
import os
from typing import Generator, Tuple, Optional
import numpy as np

class VideoStream:
    """
    Production VideoStream wrapper supporting:
    - Local video files (with optional looping for development)
    - Webcams (source=0)
    - RTSP / CCTV live camera streams
    """
    def __init__(self, source: str, loop: bool = True):
        self.source_str = str(source)
        self.loop = loop
        
        # Convert integer strings like "0" to int for webcam
        if self.source_str.isdigit():
            self.source = int(self.source_str)
            self.is_file = False
        else:
            self.source = self.source_str
            self.is_file = os.path.isfile(self.source_str)

        self.cap: Optional[cv2.VideoCapture] = None
        self._open_stream()

    def _open_stream(self) -> bool:
        """Opens or reopens the video capture source."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video source: {self.source}")
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else -1
        return True

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read_frames(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Yields (frame, frame_number, timestamp_seconds) sequentially.
        If loop=True and source is a file, restarts when EOF is reached.
        """
        frame_number = 0
        start_wall_time = time.time()

        while self.is_opened:
            ret, frame = self.cap.read()

            if not ret:
                if self.is_file and self.loop:
                    # Rewind to start of video for continuous development loop
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break

            frame_number += 1
            # Real-time elapsed timestamp
            timestamp_sec = time.time() - start_wall_time

            yield frame, frame_number, timestamp_sec

    def release(self):
        """Releases the underlying video capture hardware/file handle."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
