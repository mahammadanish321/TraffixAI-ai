import cv2
import numpy as np

def extract_plate_roi(vehicle_crop: np.ndarray, lower_ratio: float = 0.50) -> np.ndarray:
    """
    Extracts the candidate license plate region of interest (ROI).
    On road vehicles, plates are located on bumpers or tailgates in the lower half of the vehicle.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        raise ValueError("Vehicle crop is empty or invalid.")

    vh, vw = vehicle_crop.shape[:2]
    roi_y1 = int(vh * lower_ratio)
    roi_y2 = vh
    return vehicle_crop[roi_y1:roi_y2, 0:vw].copy()

def preprocess_plate_for_ocr(plate_roi: np.ndarray, upscale_factor: float = 2.0) -> np.ndarray:
    """
    Preprocesses license plate crop for OCR character extraction:
    1. Bicubic Upscaling (brings small characters to >28px height).
    2. Grayscale conversion.
    3. Bilateral Filter (smoothes dust/grain while keeping character edges sharp).
    4. CLAHE (Contrast Limited Adaptive Histogram Equalization) for localized contrast.
    """
    if plate_roi is None or plate_roi.size == 0:
        raise ValueError("Plate ROI is empty or invalid.")

    img = plate_roi
    if upscale_factor > 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * upscale_factor), int(h * upscale_factor)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Bilateral filter: d=9, sigmaColor=75, sigmaSpace=75
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE: clipLimit=2.0, tileGridSize=(8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    return enhanced
