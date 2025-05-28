# fast_omr_aligner.py
# Autor: ChatGPT · mayo 2025
#
# ▸ Alinea OMR con dos marcos (pareja completa) y
#   └─ si falla, intenta con un único rectángulo
#      probando las 4 rotaciones posibles para evitar distorsión.

import cv2
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional


class FastOMRAligner:
    # ---------- 1 · Inicialización ----------
    def __init__(
        self,
        confidence_threshold: float = 0.95,
        debug_mode: bool = False,
        debug_dir: str = "fast_debug",
        dpi: int = 300,
        min_area: int = 1000,
        area_tol: float = 0.15,   # ±15 % de diferencia de área
        pos_tol:  float = 20.0    # error medio de vértice (px)
    ):
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self.debug_dir = Path(debug_dir)
        self.dpi = dpi
        self.min_area = min_area
        self.area_tol = area_tol
        self.pos_tol = pos_tol
        self.current_id: Optional[str] = None
        if self.debug_mode:
            self._setup_logger()

    # ---------- 2 · Herramientas de debug ----------
    def _setup_logger(self):
        (self.debug_dir / "logs").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.debug_dir / "logs" / "fast_aligner.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _save_img(self, img: np.ndarray, name: str):
        if not self.debug_mode or self.current_id is None:
            return
        p = self.debug_dir / "images" / self.current_id
        p.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p / f"{name}.png"), img)

    def _save_json(self, data: dict, name: str):
        if not self.debug_mode or self.current_id is None:
            return
        p = self.debug_dir / "data" / self.current_id
        p.mkdir(parents=True, exist_ok=True)
        with open(p / f"{name}.json", "w", encoding="utf8") as f:
            json.dump(data, f, indent=2, default=str)

    # ---------- 3 · Ordenar vértices ----------
    @staticmethod
    def _order_box(box: np.ndarray) -> np.ndarray:
        pts = box.astype("float32")
        s = pts.sum(axis=1)          # x + y
        diff = np.diff(pts, axis=1)  # y – x
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    # ---------- 4 · Detectar rectángulos grandes ----------
    def _rect_candidates(self, img: np.ndarray, tag: str) -> List[Tuple[float, np.ndarray]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        self._save_img(gray, f"01_gray_{tag}")

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        self._save_img(blur, f"02_blur_{tag}")

        th = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10)
        self._save_img(th, f"03_thresh_{tag}")

        dil = cv2.dilate(th, np.ones((2, 2), np.uint8), 1)
        self._save_img(dil, f"04_dilate_{tag}")

        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dbg = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

        cands: List[Tuple[float, np.ndarray]] = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            area = w * h
            if area > self.min_area:
                box = cv2.boxPoints(rect).astype(int)
                cands.append((area, box))
                cv2.drawContours(dbg, [box], -1, (0, 255, 0), 2)

        self._save_img(dbg, f"05_contours_{tag}")
        self._save_json([{"area": a, "box": b.tolist()} for a, b in cands], f"rect_candidates_{tag}")
        return cands

    # ---------- 5 · Emparejar vértices ----------
    @staticmethod
    def _match_corners(c1, c2):
        if len(c1) < 4 or len(c2) < 4:
            return None

        A, B = np.array(c1), np.array(c2)
        cA, cB = A.mean(0), B.mean(0)
        vA, vB = A - cA, B - cB
        dA, dB = np.linalg.norm(vA, axis=1), np.linalg.norm(vB, axis=1)
        aA, aB = np.arctan2(vA[:, 1], vA[:, 0]), np.arctan2(vB[:, 1], vB[:, 0])

        matches = []
        for i, (da, aa) in enumerate(zip(dA, aA)):
            best_j, best_score = None, 1e9
            for j, (db, ab) in enumerate(zip(dB, aB)):
                score = abs(da - db) / dA.mean() + min(abs(aa - ab), 2*np.pi - abs(aa - ab))
                if score < best_score:
                    best_score, best_j = score, j
            if best_j is not None:
                matches.append((i, best_j))

        return matches if len(matches) >= 4 else None

    # ---------- 6 · Verificar homografía ----------
    @staticmethod
    def _verify_H(H, shp):
        det = abs(np.linalg.det(H))
        if not (0.7 < det < 1.3):
            return False
        h, w = shp
        pts = np.float32([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        t = H @ pts
        t /= t[2]
        return (t[0].min() > -0.1*w and t[0].max() < 1.1*w and
                t[1].min() > -0.1*h and t[1].max() < 1.1*h)

    @staticmethod
    def _rot_avg_dist(boxA, boxB):
        a = FastOMRAligner._order_box(boxA)
        b0 = FastOMRAligner._order_box(boxB)
        best = 1e9
        for k in range(4):
            b = np.roll(b0, k, axis=0)
            best = min(best, np.mean(np.linalg.norm(a - b, axis=1)))
        return best

    # ---------- 3 bis · Fallback: un solo rectángulo ----------
    def _try_single_rects(
        self,
        original: np.ndarray,
        scanned: np.ndarray,
        rects_orig, rects_scan,
        background_color=(255, 255, 255)
    ) -> Optional[np.ndarray]:
        """Intenta alinear usando un marco de cada imagen probando las 4 rotaciones."""
        K = 3  # cuántos rectángulos (más grandes) probar de cada lista
        best_aligned, best_score = None, 1e9

        def _distortion_score(H):
            A = H[:2, :2] / H[2, 2]
            shear = abs(A[0, 1]) + abs(A[1, 0])
            scale = abs(np.linalg.det(A) - 1)
            return shear + scale

        for _, boxO in rects_orig[:K]:
            dst = self._order_box(boxO)
            for _, boxS in rects_scan[:K]:
                src0 = self._order_box(boxS)

                for rot in range(4):
                    src = np.roll(src0, -rot, axis=0)
                    H = cv2.getPerspectiveTransform(src, dst)
                    if H is None or not self._verify_H(H, original.shape[:2]):
                        continue

                    score = _distortion_score(H)
                    if score < best_score:
                        best_score = score
                        best_aligned = cv2.warpPerspective(
                            scanned, H,
                            (original.shape[1], original.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=background_color
                        )
                        if self.debug_mode:
                            self.logger.debug(f"Fallback candidato rot={rot*90}°, score={score:.3f}")

        if best_aligned is not None and self.debug_mode:
            self.logger.info("Alineación exitosa con fallback (1 marco).")
            self._save_img(best_aligned, "06_aligned_fallback")

        return best_aligned

    # ---------- 7 · Alineamiento principal ----------
    def align(
        self,
        original: np.ndarray,
        scanned: np.ndarray,
        background_color=(255, 255, 255),
        debug_id: Optional[str] = None
    ) -> Optional[np.ndarray]:

        self.current_id = debug_id or f"fast_{int(time.time())}"
        if self.debug_mode:
            self.logger.info(f"=== ALIGN {self.current_id} ===")

        # 1. Detectar rectángulos
        c_orig = sorted(self._rect_candidates(original, "orig"), key=lambda x: -x[0])
        c_scan = sorted(self._rect_candidates(scanned, "scan"), key=lambda x: -x[0])
        if not c_orig or not c_scan:
            if self.debug_mode:
                self.logger.error("Sin marcos detectados.")
            return None

        # 2. Emparejar por área + posición
        used_scan, valid_pairs, dbg_pairs = set(), [], []
        for areaO, boxO in c_orig:
            best_j, best_err, best_boxS = None, 1e9, None
            for j, (areaS, boxS) in enumerate(c_scan):
                if j in used_scan:
                    continue
                err = abs(areaO - areaS) / max(areaO, areaS)
                if err < best_err:
                    best_err, best_j, best_boxS = err, j, boxS
            if best_boxS is None or best_err > self.area_tol:
                continue

            pos_err = self._rot_avg_dist(boxO, best_boxS)
            dbg_pairs.append({"area_diff": best_err, "pos_err": pos_err})

            if pos_err <= self.pos_tol:
                valid_pairs.append((boxO, best_boxS))
                used_scan.add(best_j)

        self._save_json({"pairs_debug": dbg_pairs}, "pairs_eval")

        # 3. Si no hubo pares válidos → fallback
        if not valid_pairs:
            if self.debug_mode:
                self.logger.warning("Sin pares válidos: probando fallback.")
            return self._try_single_rects(original, scanned, c_orig, c_scan, background_color)

        # 4. Reunir vértices de los pares válidos
        cornersO, cornersS = [], []
        for boxO, boxS in valid_pairs:
            cornersO.extend([(float(x), float(y)) for x, y in boxO])
            cornersS.extend([(float(x), float(y)) for x, y in boxS])

        matches = self._match_corners(cornersO, cornersS)
        if matches is None:
            if self.debug_mode:
                self.logger.error("No matches: usando fallback.")
            return self._try_single_rects(original, scanned, c_orig, c_scan, background_color)

        src = np.float32([cornersO[i] for i, _ in matches])
        dst = np.float32([cornersS[j] for _, j in matches])

        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0, maxIters=500)
        if H is None or not self._verify_H(H, original.shape[:2]):
            if self.debug_mode:
                self.logger.error("Homografía inválida: usando fallback.")
            return self._try_single_rects(original, scanned, c_orig, c_scan, background_color)

        aligned = cv2.warpPerspective(
            scanned, H,
            (original.shape[1], original.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )

        if self.debug_mode:
            self._save_img(aligned, "06_aligned")
            self._save_json({
                "pairs_used": len(valid_pairs),
                "num_matches": len(matches)
            }, "summary")

        return aligned
