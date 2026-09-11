import os
import cv2
import numpy as np
import easyocr
from typing import Tuple, Optional, List
from ultralytics import YOLO
from anpr.preprocessor import extract_plate_roi, preprocess_plate_for_ocr
from anpr.plate_parser import clean_and_correct_plate

class ANPREngine:
    """
    Automatic Number Plate Recognition (ANPR) Engine.
    
    Responsibilities:
    - Pinpoints license plate sub-bounding box using license_plate_detector.pt (or ROI extraction).
    - Enhances character contrast via bilateral filtering and CLAHE.
    - Performs deep learning OCR character recognition (EasyOCR).
    - Applies Indian registration syntax rules to correct optical ambiguities.
    """
    def __init__(self, gpu: bool = False, model_path: Optional[str] = None):
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.plate_detector: Optional[YOLO] = None
        
        default_model = model_path or os.path.join(os.path.dirname(__file__), "..", "models", "license_plate_detector.pt")
        if os.path.exists(default_model):
            try:
                self.plate_detector = YOLO(default_model)
            except Exception as e:
                print(f"[WARN] Could not load license_plate_detector in ANPREngine: {e}")

    def read_plate(self, vehicle_crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Extracts and verifies the license plate from a vehicle crop.
        
        Returns:
            (plate_number, confidence) if readable text is detected, else (None, None).
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return (None, None)

        try:
            # 1. First attempt: locate exact plate ROI with YOLO plate detector
            plate_roi = None
            if self.plate_detector is not None:
                try:
                    p_res = self.plate_detector(vehicle_crop, conf=0.20, verbose=False)[0]
                    if p_res.boxes is not None and len(p_res.boxes) > 0:
                        # Take highest conf box
                        best_pbox = max(p_res.boxes, key=lambda b: float(b.conf[0]))
                        px1, py1, px2, py2 = [max(0, int(v)) for v in best_pbox.xyxy[0].tolist()]
                        # Add small padding if within crop bounds
                        pad = 4
                        vh, vw = vehicle_crop.shape[:2]
                        px1 = max(0, px1 - pad)
                        py1 = max(0, py1 - pad)
                        px2 = min(vw, px2 + pad)
                        py2 = min(vh, py2 + pad)
                        if (px2 - px1) > 10 and (py2 - py1) > 8:
                            plate_roi = vehicle_crop[py1:py2, px1:px2].copy()
                except Exception:
                    plate_roi = None

            # Fallback to lower vehicle bumper heuristic
            if plate_roi is None or plate_roi.size == 0:
                plate_roi = extract_plate_roi(vehicle_crop)

            # 2. Preprocess (Upscale + Bilateral + CLAHE)
            preprocessed = preprocess_plate_for_ocr(plate_roi, upscale_factor=2.0)

            # 3. EasyOCR Character Extraction
            ocr_results = self.reader.readtext(
                preprocessed,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False
            )

            if not ocr_results:
                return (None, None)

            best_plate: Optional[str] = None
            best_conf: float = 0.0
            found_valid_syntax = False

            # Individual box evaluation
            for bbox, raw_text, conf in ocr_results:
                corrected, is_valid_syntax = clean_and_correct_plate(raw_text)
                if not corrected:
                    continue

                conf_float = float(conf)

                if is_valid_syntax:
                    if not found_valid_syntax or conf_float > best_conf:
                        best_plate = corrected
                        best_conf = max(conf_float, 0.85)
                        found_valid_syntax = True
                elif not found_valid_syntax and conf_float > best_conf:
                    best_plate = corrected
                    best_conf = conf_float

            # Multi-line combination (e.g. 2-row commercial plates: WB04B + 1574)
            if len(ocr_results) > 1 and not found_valid_syntax:
                sorted_boxes = sorted(ocr_results, key=lambda b: (b[0][0][1], b[0][0][0]))
                combined_raw = "".join([b[1] for b in sorted_boxes])
                combined_corrected, combined_valid = clean_and_correct_plate(combined_raw)
                if combined_corrected:
                    avg_conf = sum(float(b[2]) for b in sorted_boxes) / len(sorted_boxes)
                    if combined_valid or avg_conf > best_conf:
                        best_plate = combined_corrected
                        best_conf = max(avg_conf, 0.88 if combined_valid else avg_conf)

            if best_plate and (len(best_plate) >= 6 or best_conf > 0.25):
                return (best_plate, round(best_conf, 2))

            return (None, None)

        except Exception as err:
            print(f"[WARN] ANPR extraction failed on crop: {err}")
            return (None, None)
