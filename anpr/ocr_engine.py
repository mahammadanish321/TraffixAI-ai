import cv2
import numpy as np
import easyocr
from typing import Tuple, Optional, List
from anpr.preprocessor import extract_plate_roi, preprocess_plate_for_ocr
from anpr.plate_parser import clean_and_correct_plate

class ANPREngine:
    """
    Automatic Number Plate Recognition (ANPR) Engine (Person 2 Component).
    
    Responsibilities:
    - Isolates candidate license plate regions from vehicle crops.
    - Enhances character contrast via bilateral filtering and CLAHE.
    - Performs deep learning OCR character recognition.
    - Applies Indian registration syntax rules to correct optical ambiguities.
    """
    def __init__(self, gpu: bool = False):
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read_plate(self, vehicle_crop: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Extracts and verifies the license plate from a vehicle crop.
        
        Returns:
            (plate_number, confidence) if readable text is detected, else (None, None).
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return (None, None)

        try:
            # 1. Extract lower vehicle region where plate resides
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

            for bbox, raw_text, conf in ocr_results:
                corrected, is_valid_syntax = clean_and_correct_plate(raw_text)
                if not corrected:
                    continue

                conf_float = float(conf)

                # Prioritize detections matching valid Indian plate syntax
                if is_valid_syntax:
                    if not found_valid_syntax or conf_float > best_conf:
                        best_plate = corrected
                        best_conf = conf_float
                        found_valid_syntax = True
                elif not found_valid_syntax and conf_float > best_conf:
                    best_plate = corrected
                    best_conf = conf_float

            if best_plate and best_conf > 0.30:
                return (best_plate, round(best_conf, 2))

            return (None, None)

        except Exception as err:
            print(f"[WARN] ANPR extraction failed on crop: {err}")
            return (None, None)
