import time
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple


class FastOMRAligner:
    def __init__(self, 
                 confidence_threshold: float = 0.95,
                 debug_mode: bool = False,
                 dpi: int = 300,
                 min_area: int = 1000):
        """
        Initialize the fast OMR aligner with corner detection capabilities
        """
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self.dpi = dpi
        self.min_border_len = int(dpi * 0.5)
        self.min_area = min_area
        self.timings = {}
    
    def extract_corners(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Extract corners using minAreaRect approach."""
        t_start = time.time()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Filtro gaussiano
        processed = cv2.GaussianBlur(gray, (5, 5), 0)

        # Reduce resolution
        scale = 1
        processed = cv2.resize(processed, None, fx=scale, fy=scale)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Dilate
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by min_area (adjusted by scale)
        min_area_scaled = self.min_area * (scale ** 2)
        
        corners_mask = np.zeros_like(processed)
        corners = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
                
            if area > min_area_scaled:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                
                # Draw corners on mask
                for corner in box:
                    x, y = corner
                    cv2.circle(corners_mask, (int(x), int(y)), 5, 255, -1)
                    scaled_corner = (x / scale, y / scale)
                    corners.append(scaled_corner)

        # Scale mask back to original size
        corners_mask = cv2.resize(corners_mask, (img.shape[1], img.shape[0]))
        
        elapsed = time.time() - t_start
        self.timings['extract_corners'] = elapsed
        
        return corners_mask, corners

    def match_corners(self, 
                     corners1: List[Tuple[float, float]], 
                     corners2: List[Tuple[float, float]]) -> Optional[List[Tuple[int, int]]]:
        """Match corners between images using distance-based approach."""
        if len(corners1) < 4 or len(corners2) < 4:
            return None

        matches = []
        corners1_array = np.array(corners1)
        corners2_array = np.array(corners2)

        # Find centroid
        centroid1 = np.mean(corners1_array, axis=0)
        centroid2 = np.mean(corners2_array, axis=0)

        # Calculate distances and angles to centroid
        vectors1 = corners1_array - centroid1
        vectors2 = corners2_array - centroid2

        distances1 = np.linalg.norm(vectors1, axis=1)
        distances2 = np.linalg.norm(vectors2, axis=1)

        angles1 = np.arctan2(vectors1[:, 1], vectors1[:, 0])
        angles2 = np.arctan2(vectors2[:, 1], vectors2[:, 0])

        # Match corners based on similar relative positions
        for i, (dist1, angle1) in enumerate(zip(distances1, angles1)):
            best_match = None
            min_diff = float('inf')
            
            for j, (dist2, angle2) in enumerate(zip(distances2, angles2)):
                dist_diff = abs(dist1 - dist2)
                angle_diff = min(abs(angle1 - angle2), 2*np.pi - abs(angle1 - angle2))
                
                # Combined difference metric
                total_diff = dist_diff/np.mean(distances1) + angle_diff
                
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))

        return matches if len(matches) >= 4 else None

    def verify_homography(self, 
                         H: np.ndarray, 
                         original_shape: Tuple[int, int]) -> float:
        """Verify homography with timing."""
        t_start = time.time()
        try:
            # Check determinant
            det = np.linalg.det(H)
            if not (0.7 < abs(det) < 1.3):
                return 0.0
            
            # Verify transformation
            corners = np.array([
                [0, 0, 1],
                [original_shape[1], 0, 1],
                [original_shape[1], original_shape[0], 1],
                [0, original_shape[0], 1]
            ])
            
            transformed = H @ corners.T
            transformed = transformed / transformed[2]
            
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
        """Main alignment method using corner detection."""
        t_total = time.time()
        try:
            # Reset timings
            self.timings = {}
            
            # Extract corners
            t_extract = time.time()
            original_corners_mask, original_corners = self.extract_corners(original)
            scanned_corners_mask, scanned_corners = self.extract_corners(scanned)
            self.timings['total_extraction'] = time.time() - t_extract
            
            # Match corners
            t_match = time.time()
            matches = self.match_corners(original_corners, scanned_corners)
            self.timings['corner_matching'] = time.time() - t_match
            
            if matches is None or len(matches) < 4:
                return None
            
            # Calculate homography
            t_homography = time.time()
            src_pts = np.float32([original_corners[m[0]] for m in matches])
            dst_pts = np.float32([scanned_corners[m[1]] for m in matches])
            
            H, mask = cv2.findHomography(
                dst_pts, src_pts, 
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=500
            )
            self.timings['homography_calculation'] = time.time() - t_homography
            
            if H is None:
                return None
            
            # Verify
            if not self.verify_homography(H, original.shape[:2]):
                return None
            
            # Apply transformation
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
            return None


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
