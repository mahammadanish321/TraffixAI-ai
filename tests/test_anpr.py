import os
import sys
import time
import re
import cv2
import numpy as np
import easyocr

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── 1. Indian State/UT Codes ──
INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

# ── 2. Optical Character Confusion Matrix ──
# Maps visually ambiguous letters to digits (when a digit is required)
LETTER_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'B': '8',
    'S': '5',
    'G': '6',
    'T': '7',
    'A': '4'
}

# Maps visually ambiguous digits to letters (when a letter is required)
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '8': 'B',
    '5': 'S',
    '6': 'G',
    '4': 'A'
}

def clean_and_correct_plate(raw_text: str) -> str:
    """
    Cleans raw OCR output and applies Indian Vehicle Registration Syntax Rules:
    Format: [2 letters State][2 digits RTO][1-3 letters Series][4 digits Number]
    Example: 'MH1ZDE1432' -> 'MH12DE1432' ('Z' corrected to '2')
    """
    # Remove whitespace, hyphens, and the 'IND' watermark
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    cleaned = re.sub(r'^IND', '', cleaned)

    if len(cleaned) < 8 or len(cleaned) > 11:
        return cleaned

    chars = list(cleaned)

    # Rule 1: Positions 0 and 1 MUST be letters (State Code)
    for i in (0, 1):
        if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]

    # Rule 2: Positions 2 and 3 MUST be digits (District/RTO code)
    for i in (2, 3):
        if not chars[i].isdigit() and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    # Rule 3: The last 4 characters MUST be digits (Registration Number)
    for i in range(len(chars) - 4, len(chars)):
        if not chars[i].isdigit() and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    corrected = "".join(chars)
    return corrected

def preprocess_plate(img: np.ndarray, upscale_factor: float = 2.0) -> np.ndarray:
    """
    Preprocesses license plate crop for OCR:
    1. Upscales using bicubic interpolation if characters are small.
    2. Converts to Grayscale.
    3. Applies Bilateral Filtering (smoothes noise, preserves edges).
    4. Applies CLAHE for local contrast equalization.
    """
    if upscale_factor > 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * upscale_factor), int(h * upscale_factor)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    return enhanced

def create_sample_indian_plate(plate_number: str = "MH12DE1432") -> np.ndarray:
    """Generates a standard Indian High Security Registration Plate (HSRP) for validation."""
    plate = np.ones((80, 280, 3), dtype=np.uint8) * 255
    # Blue IND band on left
    plate[:, :40] = [200, 100, 0] # BGR
    cv2.putText(plate, 'IND', (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # Registration Text
    cv2.putText(plate, plate_number, (48, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
    return plate

def run_test():
    print("=" * 68)
    print("      TRAFFIX AI — PERSON 2: ANPR & SYNTAX ENGINE VALIDATION      ")
    print("=" * 68)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize EasyOCR
    print("[1/3] Initializing EasyOCR Reader (CPU mode)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # 2. Test with realistic Indian HSRP Plate
    print("\n[2/3] Running ANPR on Sample Indian License Plate...")
    sample_plate = create_sample_indian_plate("MH12DE1432")
    sample_path = os.path.join(output_dir, "sample_hsrp_plate.jpg")
    cv2.imwrite(sample_path, sample_plate)

    # Preprocess
    preprocessed = preprocess_plate(sample_plate)
    prep_path = os.path.join(output_dir, "sample_hsrp_preprocessed.jpg")
    cv2.imwrite(prep_path, preprocessed)

    # OCR Inference
    t0 = time.time()
    results = reader.readtext(preprocessed, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    latency_ms = (time.time() - t0) * 1000

    print(f"[INFO] OCR Latency: {latency_ms:.1f}ms on CPU")
    print("-" * 68)
    for idx, (bbox, raw_text, conf) in enumerate(results, 1):
        corrected = clean_and_correct_plate(raw_text)
        state_valid = "✅ Valid State" if corrected[:2] in INDIAN_STATES else "⚠️ Unknown State"
        print(f"Reading #{idx}:")
        print(f"  • Raw OCR Text     : '{raw_text}' (conf: {conf:.2f})")
        print(f"  • Syntax Corrected : '{corrected}' ({state_valid})")

    print("=" * 68)
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 68)

if __name__ == "__main__":
    run_test()
