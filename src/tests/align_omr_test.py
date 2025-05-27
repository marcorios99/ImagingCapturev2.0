import cv2
import numpy as np
import time
import os
from typing import Tuple, List, Optional
import logging
import json
from pathlib import Path

class FastOMRAligner:
    def __init__(self, 
                 confidence_threshold: float = 0.95,
                 debug_mode: bool = False,
                 dpi: int = 300,
                 min_area: int = 1000,
                 debug_dir: str = "debug_alignment"):
        """
        Initialize the fast OMR aligner with corner detection capabilities
        """
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self.dpi = dpi
        self.min_border_len = int(dpi * 0.5)
        self.min_area = min_area
        self.timings = {}
        self.debug_dir = debug_dir
        self.current_debug_id = None
        
        # Setup logging
        if self.debug_mode:
            self.setup_logging()
            
    def setup_logging(self):
        """Setup logging for debug mode"""
        log_dir = Path(self.debug_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'alignment_debug.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_debug_image(self, image: np.ndarray, name: str, subfolder: str = ""):
        """Save debug image with proper naming"""
        if not self.debug_mode or self.current_debug_id is None:
            return
            
        debug_path = Path(self.debug_dir) / "images" / self.current_debug_id
        if subfolder:
            debug_path = debug_path / subfolder
        debug_path.mkdir(parents=True, exist_ok=True)
        
        filename = debug_path / f"{name}.png"
        cv2.imwrite(str(filename), image)
        
        if hasattr(self, 'logger'):
            self.logger.debug(f"Saved debug image: {filename}")
    
    def save_debug_data(self, data: dict, name: str):
        """Save debug data as JSON"""
        if not self.debug_mode or self.current_debug_id is None:
            return
            
        debug_path = Path(self.debug_dir) / "data" / self.current_debug_id
        debug_path.mkdir(parents=True, exist_ok=True)
        
        filename = debug_path / f"{name}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        if hasattr(self, 'logger'):
            self.logger.debug(f"Saved debug data: {filename}")

    def extract_corners(self, img: np.ndarray, img_name: str = "") -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Extract corners using minAreaRect approach with detailed debug."""
        t_start = time.time()
        
        if self.debug_mode and hasattr(self, 'logger'):
            self.logger.info(f"=== EXTRACTING CORNERS FOR {img_name} ===")
            self.logger.info(f"Input image shape: {img.shape}")
            self.logger.info(f"Min area threshold: {self.min_area}")
        
        # Save original image
        if self.debug_mode:
            self.save_debug_image(img, f"01_original_{img_name}")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.debug_mode and hasattr(self, 'logger'):
                self.logger.debug(f"Converted to grayscale from {img.shape} to {gray.shape}")
        else:
            gray = img.copy()
            if self.debug_mode and hasattr(self, 'logger'):
                self.logger.debug("Input already grayscale")
        
        if self.debug_mode:
            self.save_debug_image(gray, f"02_grayscale_{img_name}")
        
        # Gaussian blur
        processed = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.debug_mode:
            self.save_debug_image(processed, f"03_gaussian_blur_{img_name}")
            if hasattr(self, 'logger'):
                self.logger.debug("Applied Gaussian blur (5x5)")

        # Scale (currently 1, but keeping for future use)
        scale = 1
        if scale != 1:
            processed = cv2.resize(processed, None, fx=scale, fy=scale)
            if self.debug_mode and hasattr(self, 'logger'):
                self.logger.debug(f"Scaled image by factor {scale}")
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        if self.debug_mode:
            self.save_debug_image(binary, f"04_adaptive_threshold_{img_name}")
            if hasattr(self, 'logger'):
                self.logger.debug("Applied adaptive threshold (blockSize=21, C=10)")
        
        # Dilation
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        if self.debug_mode:
            self.save_debug_image(dilated, f"05_dilated_{img_name}")
            if hasattr(self, 'logger'):
                self.logger.debug("Applied dilation (2x2 kernel, 1 iteration)")

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if self.debug_mode and hasattr(self, 'logger'):
            self.logger.info(f"Found {len(contours)} contours")

        # Debug: Draw all contours
        if self.debug_mode:
            all_contours_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
            self.save_debug_image(all_contours_img, f"06_all_contours_{img_name}")

        # Filter by area and extract corners
        min_area_scaled = self.min_area * (scale ** 2)
        corners_mask = np.zeros_like(processed)
        corners = []
        valid_contours = []
        corner_data = []
        
        if self.debug_mode:
            filtered_contours_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            if self.debug_mode:
                corner_info = {
                    'contour_id': i,
                    'area': float(area),
                    'min_area_threshold': float(min_area_scaled),
                    'passes_area_filter': area > min_area_scaled
                }
            
            if area > min_area_scaled:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                
                valid_contours.append(cnt)
                
                # Draw corners on mask and collect coordinates
                contour_corners = []
                for corner in box:
                    x, y = corner
                    cv2.circle(corners_mask, (int(x), int(y)), 5, 255, -1)
                    scaled_corner = (x / scale, y / scale)
                    corners.append(scaled_corner)
                    contour_corners.append([float(x/scale), float(y/scale)])
                
                if self.debug_mode:
                    corner_info['corners'] = contour_corners
                    corner_info['rect_center'] = [float(rect[0][0]/scale), float(rect[0][1]/scale)]
                    corner_info['rect_size'] = [float(rect[1][0]/scale), float(rect[1][1]/scale)]
                    corner_info['rect_angle'] = float(rect[2])
                    
                    # Draw on filtered contours image
                    cv2.drawContours(filtered_contours_img, [cnt], -1, (0, 255, 0), 2)
                    for corner in box:
                        cv2.circle(filtered_contours_img, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
            
            if self.debug_mode:
                corner_data.append(corner_info)

        if self.debug_mode:
            self.save_debug_image(filtered_contours_img, f"07_filtered_contours_{img_name}")
            self.save_debug_data(corner_data, f"corner_extraction_{img_name}")

        # Scale mask back to original size
        if scale != 1:
            corners_mask = cv2.resize(corners_mask, (img.shape[1], img.shape[0]))
        
        if self.debug_mode:
            self.save_debug_image(corners_mask, f"08_corners_mask_{img_name}")
            if hasattr(self, 'logger'):
                self.logger.info(f"Extracted {len(corners)} corners from {len(valid_contours)} valid contours")
        
        elapsed = time.time() - t_start
        self.timings[f'extract_corners_{img_name}'] = elapsed
        
        if self.debug_mode and hasattr(self, 'logger'):
            self.logger.info(f"Corner extraction completed in {elapsed:.4f}s")
        
        return corners_mask, corners

    def match_corners(self, 
                     corners1: List[Tuple[float, float]], 
                     corners2: List[Tuple[float, float]]) -> Optional[List[Tuple[int, int]]]:
        """Match corners between images with detailed debug."""
        t_start = time.time()
        
        if self.debug_mode and hasattr(self, 'logger'):
            self.logger.info("=== MATCHING CORNERS ===")
            self.logger.info(f"Original corners: {len(corners1)}")
            self.logger.info(f"Scanned corners: {len(corners2)}")
        
        if len(corners1) < 4 or len(corners2) < 4:
            if self.debug_mode and hasattr(self, 'logger'):
                self.logger.error(f"Insufficient corners: original={len(corners1)}, scanned={len(corners2)}")
            return None

        corners1_array = np.array(corners1)
        corners2_array = np.array(corners2)

        # Calculate centroids
        centroid1 = np.mean(corners1_array, axis=0)
        centroid2 = np.mean(corners2_array, axis=0)

        if self.debug_mode and hasattr(self, 'logger'):
            self.logger.debug(f"Centroid 1: {centroid1}")
            self.logger.debug(f"Centroid 2: {centroid2}")

        # Calculate distances and angles to centroid
        vectors1 = corners1_array - centroid1
        vectors2 = corners2_array - centroid2

        distances1 = np.linalg.norm(vectors1, axis=1)
        distances2 = np.linalg.norm(vectors2, axis=1)

        angles1 = np.arctan2(vectors1[:, 1], vectors1[:, 0])
        angles2 = np.arctan2(vectors2[:, 1], vectors2[:, 0])

        if self.debug_mode:
            matching_data = {
                'corners1': corners1,
                'corners2': corners2,
                'centroid1': centroid1.tolist(),
                'centroid2': centroid2.tolist(),
                'distances1': distances1.tolist(),
                'distances2': distances2.tolist(),
                'angles1': angles1.tolist(),
                'angles2': angles2.tolist()
            }

        # Match corners
        matches = []
        match_details = []
        
        for i, (dist1, angle1) in enumerate(zip(distances1, angles1)):
            best_match = None
            min_diff = float('inf')
            match_candidates = []
            
            for j, (dist2, angle2) in enumerate(zip(distances2, angles2)):
                dist_diff = abs(dist1 - dist2)
                angle_diff = min(abs(angle1 - angle2), 2*np.pi - abs(angle1 - angle2))
                
                # Combined difference metric
                total_diff = dist_diff/np.mean(distances1) + angle_diff
                
                match_candidates.append({
                    'candidate_idx': j,
                    'distance_diff': float(dist_diff),
                    'angle_diff': float(angle_diff),
                    'total_diff': float(total_diff)
                })
                
                if total_diff < min_diff:
                    min_diff = total_diff
                    best_match = j
            
            match_detail = {
                'original_corner_idx': i,
                'original_corner': corners1[i],
                'best_match_idx': best_match,
                'best_match_corner': corners2[best_match] if best_match is not None else None,
                'min_difference': float(min_diff),
                'all_candidates': match_candidates
            }
            match_details.append(match_detail)
            
            if best_match is not None:
                matches.append((i, best_match))

        if self.debug_mode:
            matching_data['match_details'] = match_details
            matching_data['final_matches'] = matches
            matching_data['num_matches'] = len(matches)
            self.save_debug_data(matching_data, "corner_matching")
            
            if hasattr(self, 'logger'):
                self.logger.info(f"Found {len(matches)} corner matches")
                for match in matches:
                    self.logger.debug(f"Match: original[{match[0]}] -> scanned[{match[1]}]")

        elapsed = time.time() - t_start
        self.timings['corner_matching'] = elapsed
        
        result = matches if len(matches) >= 4 else None
        if self.debug_mode and hasattr(self, 'logger'):
            if result is None:
                self.logger.error(f"Insufficient matches: {len(matches)} < 4")
            else:
                self.logger.info(f"Corner matching successful: {len(matches)} matches")
        
        return result

    def verify_homography(self, 
                         H: np.ndarray, 
                         original_shape: Tuple[int, int]) -> float:
        """Verify homography with detailed debug."""
        t_start = time.time()
        
        if self.debug_mode and hasattr(self, 'logger'):
            self.logger.info("=== VERIFYING HOMOGRAPHY ===")
            self.logger.debug(f"Homography matrix:\n{H}")
            self.logger.debug(f"Original shape: {original_shape}")
        
        verification_data = {
            'homography_matrix': H.tolist(),
            'original_shape': original_shape
        }
        
        try:
            # Check determinant
            det = np.linalg.det(H)
            verification_data['determinant'] = float(det)
            verification_data['det_valid'] = 0.7 < abs(det) < 1.3
            
            if self.debug_mode and hasattr(self, 'logger'):
                self.logger.debug(f"Determinant: {det} (valid: {verification_data['det_valid']})")
            
            if not verification_data['det_valid']:
                if self.debug_mode:
                    self.save_debug_data(verification_data, "homography_verification")
                return 0.0
            
            # Verify transformation of corners
            corners = np.array([
                [0, 0, 1],
                [original_shape[1], 0, 1],
                [original_shape[1], original_shape[0], 1],
                [0, original_shape[0], 1]
            ])
            
            transformed = H @ corners.T
            transformed = transformed / transformed[2]
            
            verification_data['original_corners'] = corners[:, :2].tolist()
            verification_data['transformed_corners'] = transformed[:2].T.tolist()
            
            # Check bounds
            x_bounds = (-0.1 * original_shape[1], 1.1 * original_shape[1])
            y_bounds = (-0.1 * original_shape[0], 1.1 * original_shape[0])
            
            x_valid = np.all(transformed[0] >= x_bounds[0]) and np.all(transformed[0] <= x_bounds[1])
            y_valid = np.all(transformed[1] >= y_bounds[0]) and np.all(transformed[1] <= y_bounds[1])
            
            verification_data['x_bounds'] = x_bounds
            verification_data['y_bounds'] = y_bounds
            verification_data['x_valid'] = bool(x_valid)
            verification_data['y_valid'] = bool(y_valid)
            verification_data['bounds_valid'] = bool(x_valid and y_valid)
            
            if self.debug_mode and hasattr(self, 'logger'):
                self.logger.debug(f"X bounds check: {x_valid} (range: {x_bounds})")
                self.logger.debug(f"Y bounds check: {y_valid} (range: {y_bounds})")
                self.logger.debug(f"Transformed corners: {transformed[:2].T}")
            
            elapsed = time.time() - t_start
            self.timings['verify_homography'] = elapsed
            
            verification_data['verification_time'] = elapsed
            verification_data['verification_passed'] = verification_data['bounds_valid']
            
            if self.debug_mode:
                self.save_debug_data(verification_data, "homography_verification")
            
            return 1.0 if verification_data['bounds_valid'] else 0.0
            
        except Exception as e:
            verification_data['error'] = str(e)
            verification_data['verification_passed'] = False
            
            if self.debug_mode:
                self.save_debug_data(verification_data, "homography_verification")
                if hasattr(self, 'logger'):
                    self.logger.error(f"Homography verification failed: {e}")
            
            return 0.0

    def align(self, 
             original: np.ndarray, 
             scanned: np.ndarray,
             background_color: Tuple[int, int, int] = (255, 255, 255),
             debug_id: str = None) -> Optional[np.ndarray]:
        """Main alignment method with comprehensive debug."""
        t_total = time.time()
        
        # Set debug ID
        if debug_id is None:
            debug_id = f"alignment_{int(time.time())}"
        self.current_debug_id = debug_id
        
        if self.debug_mode and hasattr(self, 'logger'):
            self.logger.info(f"{'='*50}")
            self.logger.info(f"STARTING ALIGNMENT - ID: {debug_id}")
            self.logger.info(f"Original shape: {original.shape}")
            self.logger.info(f"Scanned shape: {scanned.shape}")
            self.logger.info(f"Background color: {background_color}")
            self.logger.info(f"{'='*50}")
        
        alignment_data = {
            'debug_id': debug_id,
            'original_shape': original.shape,
            'scanned_shape': scanned.shape,
            'background_color': background_color,
            'parameters': {
                'confidence_threshold': self.confidence_threshold,
                'dpi': self.dpi,
                'min_area': self.min_area
            }
        }
        
        try:
            # Reset timings
            self.timings = {}
            
            # Extract corners from both images
            t_extract = time.time()
            original_corners_mask, original_corners = self.extract_corners(original, "original")
            scanned_corners_mask, scanned_corners = self.extract_corners(scanned, "scanned")
            self.timings['total_extraction'] = time.time() - t_extract
            
            alignment_data['original_corners_count'] = len(original_corners)
            alignment_data['scanned_corners_count'] = len(scanned_corners)
            
            # Match corners
            t_match = time.time()
            matches = self.match_corners(original_corners, scanned_corners)
            self.timings['corner_matching'] = time.time() - t_match
            
            alignment_data['matches_found'] = len(matches) if matches else 0
            alignment_data['matching_successful'] = matches is not None and len(matches) >= 4
            
            if matches is None or len(matches) < 4:
                alignment_data['failure_reason'] = "Insufficient corner matches"
                alignment_data['success'] = False
                
                if self.debug_mode:
                    self.save_debug_data(alignment_data, "alignment_summary")
                    if hasattr(self, 'logger'):
                        self.logger.error("Alignment failed: Insufficient corner matches")
                
                return None
            
            # Calculate homography
            t_homography = time.time()
            src_pts = np.float32([original_corners[m[0]] for m in matches])
            dst_pts = np.float32([scanned_corners[m[1]] for m in matches])
            
            alignment_data['src_points'] = src_pts.tolist()
            alignment_data['dst_points'] = dst_pts.tolist()
            
            H, mask = cv2.findHomography(
                dst_pts, src_pts, 
                cv2.RANSAC,
                ransacReprojThreshold=5.0,
                maxIters=500
            )
            self.timings['homography_calculation'] = time.time() - t_homography
            
            alignment_data['homography_computed'] = H is not None
            
            if H is None:
                alignment_data['failure_reason'] = "Homography computation failed"
                alignment_data['success'] = False
                
                if self.debug_mode:
                    self.save_debug_data(alignment_data, "alignment_summary")
                    if hasattr(self, 'logger'):
                        self.logger.error("Alignment failed: Homography computation failed")
                
                return None
            
            alignment_data['homography_matrix'] = H.tolist()
            alignment_data['inlier_mask'] = mask.tolist() if mask is not None else None
            
            # Verify homography
            verification_score = self.verify_homography(H, original.shape[:2])
            alignment_data['verification_score'] = verification_score
            alignment_data['verification_passed'] = verification_score > 0
            
            if verification_score == 0:
                alignment_data['failure_reason'] = "Homography verification failed"
                alignment_data['success'] = False
                
                if self.debug_mode:
                    self.save_debug_data(alignment_data, "alignment_summary")
                    if hasattr(self, 'logger'):
                        self.logger.error("Alignment failed: Homography verification failed")
                
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
            
            # Save final result
            if self.debug_mode:
                self.save_debug_image(aligned, "09_final_aligned")
                
                # Create comparison image
                comparison = np.hstack([
                    cv2.resize(original, (original.shape[1]//2, original.shape[0]//2)),
                    cv2.resize(aligned, (aligned.shape[1]//2, aligned.shape[0]//2))
                ])
                self.save_debug_image(comparison, "10_comparison")
            
            # Record success
            total_time = time.time() - t_total
            self.timings['total_alignment'] = total_time
            
            alignment_data['success'] = True
            alignment_data['total_time'] = total_time
            alignment_data['timings'] = self.timings
            
            if self.debug_mode:
                self.save_debug_data(alignment_data, "alignment_summary")
                if hasattr(self, 'logger'):
                    self.logger.info(f"Alignment successful in {total_time:.4f}s")
                    self.logger.info(f"Debug files saved in: {self.debug_dir}/images/{debug_id}")
            
            return aligned
                
        except Exception as e:
            alignment_data['success'] = False
            alignment_data['failure_reason'] = f"Exception: {str(e)}"
            alignment_data['exception_type'] = type(e).__name__
            
            if self.debug_mode:
                self.save_debug_data(alignment_data, "alignment_summary")
                if hasattr(self, 'logger'):
                    self.logger.error(f"Alignment failed with exception: {e}")
            
            return None

def test_alignment(original_path: str, scanned_path: str, output_dir: str = "alignment_test"):
    """
    Función de prueba para alinear imágenes con debug completo
    """
    # Crear directorios
    Path(output_dir).mkdir(exist_ok=True)
    input_dir = Path(output_dir) / "input_images"
    input_dir.mkdir(exist_ok=True)
    
    # Cargar imágenes
    print(f"Cargando imagen original: {original_path}")
    original = cv2.imread(original_path)
    if original is None:
        print(f"Error: No se pudo cargar la imagen original: {original_path}")
        return None
    
    print(f"Cargando imagen escaneada: {scanned_path}")
    scanned = cv2.imread(scanned_path)
    if scanned is None:
        print(f"Error: No se pudo cargar la imagen escaneada: {scanned_path}")
        return None
    
    # Copiar imágenes de entrada al directorio de debug
    cv2.imwrite(str(input_dir / "original_input.png"), original)
    cv2.imwrite(str(input_dir / "scanned_input.png"), scanned)
    
    # Crear aligner con debug activado
    aligner = FastOMRAligner(
        debug_mode=True,
        debug_dir=output_dir,
        confidence_threshold=0.95,
        min_area=1000
    )
    
    # Ejecutar alineación
    debug_id = f"test_{int(time.time())}"
    print(f"\nIniciando alineación con ID: {debug_id}")
    print(f"Los archivos de debug se guardarán en: {output_dir}")
    
    result = aligner.align(original, scanned, debug_id=debug_id)
    
    if result is not None:
        # Guardar resultado final
        result_path = Path(output_dir) / "final_result.png"
        cv2.imwrite(str(result_path), result)
        print(f"\n✅ Alineación exitosa!")
        print(f"Resultado guardado en: {result_path}")
        
        # Mostrar tiempos
        print(f"\n⏱️  Tiempos de ejecución:")
        for step, timing in aligner.timings.items():
            print(f"  {step}: {timing:.4f}s")
            
    else:
        print(f"\n❌ Alineación fallida!")
        print(f"Revisa los archivos de debug en: {output_dir}")
    
    print(f"\n📁 Estructura de archivos generada:")
    print(f"  {output_dir}/")
    print(f"  ├── input_images/          # Imágenes de entrada")
    print(f"  ├── images/{debug_id}/     # Imágenes de debug paso a paso")
    print(f"  ├── data/{debug_id}/       # Datos JSON de debug")
    print(f"  ├── logs/                  # Logs detallados")
    print(f"  └── final_result.png       # Resultado final (si exitoso)")
    
    return result

# Ejemplo de uso
if __name__ == "__main__":
    # Cambiar estas rutas por las de tus imágenes
    original_image_path = "ruta/a/tu/imagen_original.png"
    scanned_image_path = "ruta/a/tu/imagen_escaneada.png"
    
    # Ejecutar prueba
    result = test_alignment(
        original_path=original_image_path,
        scanned_path=scanned_image_path,
        output_dir="debug_alignment_test"
    )