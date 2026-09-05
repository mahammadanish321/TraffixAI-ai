import re
from typing import Tuple

# All official 2-letter State & Union Territory codes in India
INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

# Optical Character Confusion Matrix (when a DIGIT is expected)
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

# Optical Character Confusion Matrix (when a LETTER is expected)
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '8': 'B',
    '5': 'S',
    '6': 'G',
    '4': 'A'
}

def clean_and_correct_plate(raw_text: str) -> Tuple[str, bool]:
    """
    Sanitizes raw OCR output and applies Indian Vehicle Registration Syntax Rules:
    Format: [2 letters State][2 digits District/RTO][1-3 letters Series][4 digits Number]
    Example: 'MH1ZDE1432' -> ('MH12DE1432', True)
    
    Returns:
        (corrected_plate_number, is_valid_indian_format)
    """
    if not raw_text:
        return ("", False)

    # 1. Sanitize: uppercase, strip whitespace, punctuation, and 'IND' watermark
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    cleaned = re.sub(r'^IND', '', cleaned)

    # Indian plates are between 8 and 11 alphanumeric characters
    if len(cleaned) < 8 or len(cleaned) > 11:
        return (cleaned, False)

    chars = list(cleaned)

    # Rule 1: Positions 0 and 1 MUST be letters (State Code)
    for i in (0, 1):
        if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]

    # Rule 2: Positions 2 and 3 MUST be digits (District/RTO code)
    for i in (2, 3):
        if not chars[i].isdigit() and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    # Rule 3: The last 4 characters MUST be digits (Unique Registration Number)
    for i in range(len(chars) - 4, len(chars)):
        if not chars[i].isdigit() and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    corrected = "".join(chars)

    # Validate against known state codes
    state_code = corrected[:2]
    is_valid_state = state_code in INDIAN_STATES

    # Standard Indian Plate Regex: ^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$
    is_valid_syntax = bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$', corrected))
    is_valid = is_valid_state and is_valid_syntax

    return (corrected, is_valid)
