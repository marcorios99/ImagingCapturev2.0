from enum import Enum, auto
from typing import Set

class ProcessingStatus(Enum):
    """Estados posibles del procesamiento de imágenes."""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()

class RegionType(Enum):
    OMR = "OMR"
    ICR = "ICR"
    BARCODE = "BARCODE"
    XMARK = "XMARK"


ALLOWED_IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
MIN_IMAGE_DPI: int = 300

# ICR constants
INDEX_TO_CLASS = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', 
    7: 'H', 
    8: 'I', 
    9: 'J',
    10: 'K', 
    11: 'L', 
    12: 'M', 
    13: 'N', 
    14: 'O', 
    15: 'P', 
    16: 'Q', 
    17: 'R', 
    18: 'S',
    19: 'T', 
    20: 'U', 
    21: 'V', 
    22: 'W', 
    23: 'X', 
    24: 'Y', 
    25: 'Z', 
    26: 'Ñ'
}



class ProcessingStatus(Enum):
    """Estados posibles del procesamiento de imágenes."""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()

class ProcessingSpeed(Enum):
    FULL = "full"
    MODERATE = "moderate"
    MANUAL = "manual"

ALLOWED_IMAGE_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}