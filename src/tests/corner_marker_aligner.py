# corner_marker_aligner.py
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Optional


class CornerMarkerAligner:
    """
    Alineador que trabaja solo con los marcadores-esquina cuadrados.
    Con `debug_mode=True` guarda las imágenes intermedias:

    ├─ <debug_dir>/
       ├─ images/<ID>/01_gray.png
       ├─ images/<ID>/02_thresh.png
       ├─ images/<ID>/03_markers.png   (marcadores dibujados en la escaneada)
       └─ images/<ID>/04_aligned.png   (resultado final)
    """

    # ---------------------------------------------------------------
    def __init__(self,
                 min_marker_area: int = 400,
                 max_marker_area: int = 8000,
                 aspect_tol: float = 0.25,
                 debug_mode: bool = False,
                 debug_dir: str = "corner_debug"):
        self.min_marker_area = min_marker_area
        self.max_marker_area = max_marker_area
        self.aspect_tol = aspect_tol
        self.debug_mode = debug_mode
        self.debug_dir = Path(debug_dir)
        self.current_id: Optional[str] = None

    # ---------- utilidades debug -----------------------------------
    def _save_img(self, img: np.ndarray, name: str):
        if not self.debug_mode or self.current_id is None:
            return
        p = self.debug_dir / "images" / self.current_id
        p.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p / f"{name}.png"), img)

    # ---------------------------------------------------------------
    def _detect_markers(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Devuelve la lista de centros (x, y) de marcadores cuadrados.
        Si está en modo debug, guarda las fases principales.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        self._save_img(gray, "01_gray")

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        th = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 31, 10)
        dil = cv2.dilate(th, np.ones((3, 3), np.uint8), 1)
        self._save_img(dil, "02_thresh")

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

        # overlay para visualización
        if self.debug_mode:
            dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for c in markers:
                cv2.circle(dbg, tuple(c.astype(int)), 12, (0, 0, 255), 2)
            self._save_img(dbg, "03_markers")

        return markers

    # ---------------------------------------------------------------
    @staticmethod
    def _order_corners(pts: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(pts) < 4:
            return None
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
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

        # id de sesión debug
        if self.debug_mode:
            self.current_id = f"corner_{int(time.time())}"

        # 1. detectar marcadores
        mk_orig = self._detect_markers(original)
        mk_scan = self._detect_markers(scanned)

        if len(mk_orig) < 4 or len(mk_scan) < 4:
            return None

        # 2. ordenar y seleccionar 4
        src = self._order_corners(sorted(mk_orig, key=lambda p: (p[1], p[0]))[:4])
        dst = self._order_corners(sorted(mk_scan, key=lambda p: (p[1], p[0]))[:4])
        if src is None or dst is None:
            return None

        # 3. homografía directa
        H = cv2.getPerspectiveTransform(dst, src)
        if H is None or not self._verify_H(H, original.shape[:2]):
            return None

        # 4. warp final
        aligned = cv2.warpPerspective(
            scanned, H,
            (original.shape[1], original.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )
        self._save_img(aligned, "04_aligned")
        return aligned
