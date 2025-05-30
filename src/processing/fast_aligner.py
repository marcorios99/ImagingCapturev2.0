import time
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
from typing import Tuple, List, Optional


class FastOMRAligner:
    """
    Alineador rápido de formularios OMR basado en detección de rectángulos.

    · Detecta los rectángulos más grandes en ambas imágenes con minAreaRect.  
    · Empareja los marcos cuyas áreas coinciden dentro de una tolerancia.  
    · Si el emparejamiento múltiple falla, intenta alinear con un único
      rectángulo por imagen, probando las cuatro rotaciones posibles para
      evitar distorsiones.  
    """

    # -----------------------------------------------------------------
    def __init__(self,
                 confidence_threshold: float = 0.95,
                 dpi: int = 300,
                 min_area: int = 900000,
                 area_tol: float = 0.15):          # ±15 % de diferencia
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi
        self.min_border_len = int(dpi * 0.5)
        self.min_area = min_area
        self.area_tol = area_tol

    # -----------------------------------------------------------------
    @staticmethod
    def _order_box(box: np.ndarray) -> np.ndarray:
        """Devuelve los vértices en orden TL-TR-BR-BL."""
        pts = box.astype("float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    # -----------------------------------------------------------------
    @staticmethod
    def _rect_candidates(img: np.ndarray,
                         min_area: int,
                         scale: float = 1.0) -> List[Tuple[float, np.ndarray]]:
        """
        Devuelve [(area, boxPts), ...] de todos los contornos con área suficiente.
        `boxPts` es un array (4, 2) con int32.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        proc = cv2.GaussianBlur(gray, (5, 5), 0)
        proc = cv2.resize(proc, None, fx=scale, fy=scale)

        th = cv2.adaptiveThreshold(proc, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
        dil = cv2.dilate(th, np.ones((2, 2), np.uint8), 1)
        contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        min_area_scaled = min_area * (scale ** 2)
        cands: List[Tuple[float, np.ndarray]] = []

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            rect_area = w * h
            if rect_area > min_area_scaled:
                box = cv2.boxPoints(rect).astype(int)
                cands.append((rect_area, box))

        return cands

    # -----------------------------------------------------------------
    @staticmethod
    def _match_corners(c1: List[Tuple[float, float]],
                       c2: List[Tuple[float, float]]
                       ) -> Optional[List[Tuple[int, int]]]:
        """Empareja esquinas por distancia y ángulo al centro."""
        if len(c1) < 4 or len(c2) < 4:
            return None
        A, B = np.array(c1), np.array(c2)
        centA, centB = A.mean(0), B.mean(0)
        vA, vB = A - centA, B - centB
        dA, dB = np.linalg.norm(vA, axis=1), np.linalg.norm(vB, axis=1)
        angA, angB = np.arctan2(vA[:, 1], vA[:, 0]), np.arctan2(vB[:, 1], vB[:, 0])

        matches = []
        for i, (da, aa) in enumerate(zip(dA, angA)):
            best, mdiff = None, 1e9
            for j, (db, ab) in enumerate(zip(dB, angB)):
                diff = abs(da - db) / dA.mean() + \
                       min(abs(aa - ab), 2*np.pi - abs(aa - ab))
                if diff < mdiff:
                    mdiff, best = diff, j
            if best is not None:
                matches.append((i, best))
        return matches if len(matches) >= 4 else None

    # -----------------------------------------------------------------
    @staticmethod
    def _verify_homography(H: np.ndarray, shape: Tuple[int, int]) -> bool:
        det = abs(np.linalg.det(H))
        if not (0.7 < det < 1.3):
            return False
        h, w = shape
        pts = np.float32([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        t = H @ pts
        t /= t[2]
        return (
            t[0].min() > -0.1 * w and t[0].max() < 1.1 * w and
            t[1].min() > -0.1 * h and t[1].max() < 1.1 * h
        )

    # -----------------------------------------------------------------
    def _try_single_rects(self,
                          original: np.ndarray,
                          scanned: np.ndarray,
                          rects_orig, rects_scan,
                          background_color=(255, 255, 255)
                          ) -> Optional[np.ndarray]:
        """
        Fallback: alinear con un único rectángulo,
        probando las 4 rotaciones de la imagen escaneada.
        """
        K = 3                     # nº de marcos más grandes a probar
        best_align, best_score = None, 1e9

        def distortion(H):
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
                    if H is None or not self._verify_homography(H, original.shape[:2]):
                        continue
                    score = distortion(H)
                    if score < best_score:
                        best_score = score
                        best_align = cv2.warpPerspective(
                            scanned, H,
                            (original.shape[1], original.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=background_color
                        )
        return best_align

    # -----------------------------------------------------------------
    def align(self,
              original: np.ndarray,
              scanned: np.ndarray,
              background_color: Tuple[int, int, int] = (255, 255, 255)
              ) -> Optional[np.ndarray]:
        """
        · Detecta todos los rectángulos grandes en ambas imágenes.  
        · Empareja los que tienen un área relativa similar (± area_tol).  
        · Usa los vértices de los pares válidos para estimar la homografía.  
        · Si falla, aplica el fallback de un solo rectángulo.
        """
        # 1. Rectángulos candidatos
        cands_orig = self._rect_candidates(original, self.min_area)
        cands_scan = self._rect_candidates(scanned,  self.min_area)
        if not cands_orig or not cands_scan:
            return None

        # 2. Buscar pares con áreas similares
        valid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        best_pair, best_err = None, 1e9
        for areaO, boxO in cands_orig:
            for areaS, boxS in cands_scan:
                rel_diff = abs(areaO - areaS) / max(areaO, areaS)
                if rel_diff <= self.area_tol:
                    valid_pairs.append((boxO, boxS))
                elif rel_diff < best_err:
                    best_err, best_pair = rel_diff, (boxO, boxS)

        if not valid_pairs and best_pair:
            valid_pairs = [best_pair]
        if not valid_pairs:
            return None

        # 3. Reunir vértices de todos los pares
        cornersO, cornersS = [], []
        for boxO, boxS in valid_pairs:
            cornersO.extend([(float(x), float(y)) for x, y in boxO])
            cornersS.extend([(float(x), float(y)) for x, y in boxS])

        # 4. Emparejar vértices y calcular homografía
        matches = self._match_corners(cornersO, cornersS)
        if matches:
            src = np.float32([cornersO[i] for i, _ in matches])
            dst = np.float32([cornersS[j] for _, j in matches])
            H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0, maxIters=500)
            if H is not None and self._verify_homography(H, original.shape[:2]):
                return cv2.warpPerspective(
                    scanned, H,
                    (original.shape[1], original.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=background_color
                )

        # 5. Fallback: un solo rectángulo
        return self._try_single_rects(original, scanned,
                                      cands_orig, cands_scan,
                                      background_color)


class FastICRAligner:
    def __init__(self, 
                 confidence_threshold: float = 0.95,
                 dpi: int = 300,
                 min_area: int = 1000,
                 max_area: int = 5000):  # Añadido parámetro max_area
        """
        Initialize the fast OMR aligner with corner detection capabilities
        """
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi
        self.min_border_len = int(dpi * 0.5)
        self.min_area = min_area
        self.max_area = max_area  # Nueva propiedad
        self.timings = {}
                
    def extract_corners(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[Tuple[float, float]]]]:
        """Extrae esquinas detectando únicamente cuadraditos negros en las cuatro esquinas de la imagen."""
        t_start = time.time()
        
        # Convertir a escala de grises
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Reducir resolución y aplicar filtro gaussiano
        processed = cv2.GaussianBlur(gray, (5, 5), 0)
        scale = 1
        processed = cv2.resize(processed, None, fx=scale, fy=scale)
        
        # Umbralización binaria inversa (para obtener cuadraditos negros)
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
        # Dilatación para cerrar huecos
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
    
        # Definir regiones de las esquinas
        h, w = dilated.shape[:2]
        corner_size = 0.2  # Ajustar este valor según sea necesario
        corner_w, corner_h = int(w * corner_size), int(h * corner_size)
        
        corners_coords = {
            'top_left': (slice(0, corner_h), slice(0, corner_w)),
            'top_right': (slice(0, corner_h), slice(w - corner_w, w)),
            'bottom_left': (slice(h - corner_h, h), slice(0, corner_w)),
            'bottom_right': (slice(h - corner_h, h), slice(w - corner_w, w))
        }
        
        corners_dict = { 'top_left': [], 'top_right': [], 'bottom_left': [], 'bottom_right': [] }
        
        for corner_name, (yslice, xslice) in corners_coords.items():
            # Extraer la región de la esquina
            corner_region = dilated[yslice, xslice]
            # Encontrar contornos en la región de la esquina
            contours, _ = cv2.findContours(
                corner_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Ajustar las coordenadas del contorno a la imagen completa
            x_offset = xslice.start
            y_offset = yslice.start
            
            # Procesar contornos
            for cnt in contours:
                area = cv2.contourArea(cnt)
                min_area_scaled = self.min_area * (scale ** 2)
                max_area_scaled = self.max_area * (scale ** 2)
                if min_area_scaled <= area <= max_area_scaled:
                    # Aproximar el contorno a un polígono
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    
                    # Verificar si es un cuadrilátero
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        # Calcular el aspecto del contorno
                        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(approx)
                        aspect_ratio = float(w_rect) / h_rect
                        if 0.8 <= aspect_ratio <= 1.2:  # Ajustar tolerancia según sea necesario
                            # Ajustar posición del contorno y escalar al tamaño original
                            approx_full = (approx + np.array([[[x_offset, y_offset]]])) / scale
                            
                            # Definir el punto de referencia de la esquina
                            image_h, image_w = img.shape[:2]
                            corner_points = {
                                'top_left': np.array([0, 0]),
                                'top_right': np.array([image_w, 0]),
                                'bottom_left': np.array([0, image_h]),
                                'bottom_right': np.array([image_w, image_h])
                            }
                            ref_point = corner_points[corner_name]
                            
                            # Encontrar el punto de la aproximación más cercano a la esquina de la imagen
                            distances = [np.linalg.norm(pt[0] - ref_point) for pt in approx_full]
                            min_index = np.argmin(distances)
                            closest_point = approx_full[min_index][0]
                            
                            # Agregar este punto al diccionario
                            corners_dict[corner_name].append(tuple(closest_point))
        
        elapsed = time.time() - t_start
        self.timings['extract_corners'] = elapsed
        
        return None, corners_dict  # Retornamos None en lugar de corners_mask, ya que no lo utilizamos
    
    def match_corners(self, 
                      corners1: Dict[str, List[Tuple[float, float]]], 
                      corners2: Dict[str, List[Tuple[float, float]]]) -> Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """Empareja las esquinas entre imágenes basándose en los nombres de las esquinas."""
        corner_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        matches = []
        for corner_name in corner_order:
            if corner_name in corners1 and corner_name in corners2:
                if len(corners1[corner_name]) == 0 or len(corners2[corner_name]) == 0:
                    return None
                # Tomar el primer punto encontrado en cada esquina
                pt1 = corners1[corner_name][0]
                pt2 = corners2[corner_name][0]
                matches.append((pt1, pt2))
            else:
                return None
        return matches

    def verify_homography(self, 
                          H: np.ndarray, 
                          original_shape: Tuple[int, int]) -> float:
        """Verifica la homografía con temporización."""
        t_start = time.time()
        try:
            # Comprobar determinante
            det = np.linalg.det(H)
            if not (0.1 < abs(det) < 10):  # Ajustado para mayor tolerancia
                return 0.0
            
            # Verificar transformación
            corners = np.array([
                [0, 0, 1],
                [original_shape[1], 0, 1],
                [original_shape[1], original_shape[0], 1],
                [0, original_shape[0], 1]
            ]).T  # Transponer para multiplicación
            
            transformed = H @ corners
            transformed /= transformed[2]  # Normalizar coordenadas homogéneas
            
            # Verificar que los puntos transformados estén dentro de un rango razonable
            if not (np.all(transformed[0] >= -0.1 * original_shape[1]) and 
                   np.all(transformed[0] <= 1.1 * original_shape[1]) and
                   np.all(transformed[1] >= -0.1 * original_shape[0]) and 
                   np.all(transformed[1] <= 1.1 * original_shape[0])):
                return 0.0
            
            elapsed = time.time() - t_start
            self.timings['verify_homography'] = elapsed
            
            return 1.0
                
        except Exception as e:
            return 0.0

    def align(self, 
              original: np.ndarray, 
              scanned: np.ndarray,
              background_color: Tuple[int, int, int] = (255, 255, 255)) -> Optional[np.ndarray]:
        """Método principal de alineamiento usando detección de esquinas."""
        t_total = time.time()
        try:
            # Reiniciar tiempos
            self.timings = {}
            
            # Extraer esquinas
            t_extract = time.time()
            _, original_corners = self.extract_corners(original)
            _, scanned_corners = self.extract_corners(scanned)
            self.timings['total_extraction'] = time.time() - t_extract
            
            # Emparejar esquinas
            t_match = time.time()
            matches = self.match_corners(original_corners, scanned_corners)
            self.timings['corner_matching'] = time.time() - t_match
            
            if matches is None or len(matches) < 4:
                #print("No se encontraron suficientes esquinas.")
                return None
            
            # Calcular homografía
            t_homography = time.time()
            src_pts = np.float32([m[0] for m in matches])  # Puntos de la original
            dst_pts = np.float32([m[1] for m in matches])  # Puntos de la escaneada
            
            H, mask = cv2.findHomography(
                dst_pts, src_pts, 
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=500
            )
            self.timings['homography_calculation'] = time.time() - t_homography
            
            if H is None:
                #print("No se pudo calcular la homografía.")
                return None
            
            # Verificar homografía
            if not self.verify_homography(H, original.shape[:2]):
                #print("La homografía no es válida.")
                return None
            
            # Aplicar transformación
            t_warp = time.time()
            aligned = cv2.warpPerspective(
                scanned,
                H,
                (original.shape[1], original.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=background_color
            )
            self.timings['warp_perspective'] = time.time() - t_warp
            
            return aligned
                
        except Exception as e:
            #print(f"Error en el alineamiento: {str(e)}")
            return None
