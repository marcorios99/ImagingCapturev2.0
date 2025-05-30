# corner_aligner_omr.py
import cv2
import numpy as np
from typing import Tuple, List, Optional


class CornerMarkerAligner:
    """
    Alineador que usa únicamente los marcadores cuadrados impresos
    en las cuatro esquinas del formulario.
    """

    # ---------------------------------------------------------------
    def __init__(self,
                 min_marker_area: int = 400,
                 max_marker_area: int = 8000,
                 aspect_tol: float = 0.25):
        self.min_marker_area = min_marker_area
        self.max_marker_area = max_marker_area
        self.aspect_tol = aspect_tol

    # ---------------------------------------------------------------
    def _detect_markers(self, img: np.ndarray) -> List[np.ndarray]:
        """Devuelve la lista de centros (x, y) de marcadores cuadrados."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        th = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 31, 10)
        dil = cv2.dilate(th, np.ones((3, 3), np.uint8), 1)

        contours, _ = cv2.findContours(
            dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if not (self.min_marker_area <= area <= self.max_marker_area):
                continue
            if abs(w / h - 1) > self.aspect_tol:
                continue
            cx, cy = x + w / 2, y + h / 2
            markers.append(np.array([cx, cy], dtype=np.float32))

        return markers

    # ---------------------------------------------------------------
    @staticmethod
    def _order_corners(pts: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(pts) < 4:
            return None
        pts = np.array(pts)
        s = pts.sum(axis=1)             # x + y
        diff = np.diff(pts, axis=1)     # y − x
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    # ---------------------------------------------------------------
    @staticmethod
    def _verify_H(H: np.ndarray, shape: Tuple[int, int]) -> bool:
        det = abs(np.linalg.det(H))
        if not (0.7 < det < 1.3):
            return False
        h, w = shape
        pts = np.float32([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        t = H @ pts;  t /= t[2]
        return (t[0].min() > -0.1*w and t[0].max() < 1.1*w and
                t[1].min() > -0.1*h and t[1].max() < 1.1*h)

    # ---------------------------------------------------------------
    def align(self,
              original: np.ndarray,
              scanned: np.ndarray,
              background_color: Tuple[int, int, int] = (255, 255, 255)
              ) -> Optional[np.ndarray]:

        mk_orig = self._detect_markers(original)
        mk_scan = self._detect_markers(scanned)
        if len(mk_orig) < 4 or len(mk_scan) < 4:
            return None

        src = self._order_corners(sorted(mk_orig, key=lambda p: (p[1], p[0]))[:4])
        dst = self._order_corners(sorted(mk_scan, key=lambda p: (p[1], p[0]))[:4])
        if src is None or dst is None:
            return None

        H = cv2.getPerspectiveTransform(dst, src)
        if H is None or not self._verify_H(H, original.shape[:2]):
            return None

        return cv2.warpPerspective(
            scanned, H,
            (original.shape[1], original.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )
