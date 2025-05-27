import time
import logging
import cv2
import numpy as np
import zxingcpp
from typing import Optional, Dict, Any, List, Tuple, NamedTuple, Union
from pathlib import Path
from dataclasses import dataclass
from core.exceptions import ProcessingError
from database.service import DatabaseService
from processing.fast_aligner import FastOMRAligner, FastICRAligner

# Configuración de logging
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Resultado del procesamiento de una región."""
    value: Any
    confidence: Optional[float]
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }

class RegionCoordinates(NamedTuple):
    """Coordenadas de una región de interés."""
    x: float
    y: float
    width: float
    height: float

    def to_pixels(self, dpi: int, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convierte coordenadas relativas a píxeles absolutos."""
        x1 = int(self.x * dpi)
        y1 = int(self.y * dpi)
        x2 = int((self.x + self.width) * dpi)
        y2 = int((self.y + self.height) * dpi)
        
        # Validar límites
        x1 = max(0, min(x1, image_shape[1]))
        y1 = max(0, min(y1, image_shape[0]))
        x2 = max(0, min(x2, image_shape[1]))
        y2 = max(0, min(y2, image_shape[0]))
        
        return x1, y1, x2, y2

class ImageProcessor:
    """Procesador principal de imágenes OMR/ICR."""
    
    def __init__(self, config: Any):
        """Inicializa el procesador con configuración."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuración de procesamiento
        self.dpi = getattr(config, 'dpi', 300)
        
        # Inicializar ICR Processor
        #models_path = Path(__file__).parent.parent / 'ml_models'
        #self.icr_processor = ICRProcessor(models_path)
        
        # Cache para plantillas de referencia
        self._reference_templates: Dict[str, np.ndarray] = {}
            
    def process_image(self, 
                    image_path: Union[str, Path],
                    barcode_regions: List[Dict[str, float]],
                    reference_paths: List[Path] = None,
                    db_service: DatabaseService = None,
                    reference_index: int = None,
                    prefix: Optional[str] = None,
                    page_number: Optional[int] = None) -> Dict[str, Any]:
        try:
            processing_start = time.time()
            
            # Cargar imagen
            image = cv2.imread(str(image_path))
            if image is None:
                raise ProcessingError(f"No se pudo cargar la imagen: {image_path}")
            
            # Preparar resultado base
            result = {
                'status': 'success',
                'barcode': None,
                'prefix': None,
                'page_number': None,
                'is_first_page': None,
                'path': str(image_path),
                'aligned': False,
                'reference_template': None,
                'processing_time': 0,
                'results': [],
                'error': None
            }
            
            # Leer código de barras
            barcode = self.read_barcode(image, barcode_regions)
            
            if barcode:
                if not self.validate_barcode(barcode):
                    self.logger.warning(f"Código de barras inválido: {barcode}")
                    result['error'] = f"Formato de código inválido: {barcode}"
                else:
                    result.update({
                        'barcode': barcode,
                        'prefix': self.get_prefix(barcode),
                        'page_number': self.get_page_number(barcode),
                        'is_first_page': self.is_first_page(barcode)
                    })
            elif prefix:
                self.logger.debug(f"Usando prefix proporcionado: {prefix}")
                result.update({
                    'barcode': None,
                    'prefix': prefix,
                    'is_first_page': None
                })
            else:
                self.logger.warning(f"No se pudo detectar código de barras y no se proporcionó prefix en la imagen {image_path}")
                result['error'] = "No se pudo detectar código de barras y no se proporcionó prefix"
            
            # Alinear imagen con referencia correspondiente
            aligned_image = image
            if reference_paths and reference_index is not None:
                try:
                    if 0 <= reference_index < len(reference_paths):
                        reference_path = reference_paths[reference_index]
                        
                        if reference_path.exists():
                            reference = self._get_reference_template(reference_path)
                            
                            if reference is not None:
                                # Obtener tipo de campo antes de alinear
                                field_type = None
                                if prefix and db_service:
                                    try:
                                        fields = db_service.get_campos_plantilla(
                                            identificador=prefix,
                                            by_prefix=True,
                                            cache=True
                                        )
                                        if fields:
                                            page_fields = [f for f in fields if f['pagina'] == reference_index + 1]
                                            if page_fields:
                                                field_type = page_fields[0]['tipo_campo']
                                    except Exception as e:
                                        self.logger.error(f"Error getting field type: {e}")
                                
                                # Seleccionar alineador según tipo de campo
                                if field_type == 'XMARK':
                                    aligner = FastICRAligner(
                                        dpi=self.dpi,
                                        min_area=200,
                                        max_area=4300
                                    )
                                else:
                                    aligner = FastOMRAligner(
                                        dpi=self.dpi,
                                        min_area=getattr(self.config, 'min_area', 900000)
                                    )
                                    
                                aligned = aligner.align(reference, image)
                                if aligned is not None:
                                    aligned_image = aligned
                                    result['aligned'] = True
                                    result['reference_index'] = reference_index
                                    result['reference_template'] = str(reference_path)
                                    self.logger.debug(f"Imagen alineada correctamente usando referencia {reference_index + 1}")
                                else:
                                    self.logger.warning(f"Alineación fallida para imagen {image_path}")
                            else:
                                self.logger.warning(f"No se pudo cargar el template de referencia: {reference_path}")
                        else:
                            self.logger.warning(f"Referencia no existe: {reference_path}")
                    else:
                        self.logger.warning(f"Índice de referencia {reference_index} fuera de rango")
                except Exception as align_error:
                    self.logger.error(f"Error en alineación: {align_error}")
                    result['error'] = str(align_error)
            
            # Remover marcas rosas
            cleaned_image = self.remove_pink_marks(aligned_image)

            # Usar el reference_index + 1 como número de página para los campos
            field_page_number = reference_index + 1 if reference_index is not None else None
            result['field_page_number'] = field_page_number

            # Procesar campos si existen
            if db_service is not None and field_page_number is not None:
                try:
                    prefix = result.get('prefix')
                    if prefix:
                        fields = db_service.get_campos_plantilla(
                            identificador=prefix,
                            by_prefix=True,
                            cache=True
                        )
                        
                        if fields:
                            # Filtrar campos usando field_page_number
                            page_fields = [
                                f for f in fields 
                                if f['pagina'] == field_page_number
                            ]
                            
                            if page_fields:
                                field_results = self.process_fields(cleaned_image, page_fields)
                                result['results'] = field_results
                                self.logger.debug(
                                    f"Procesados {len(field_results)} campos "
                                    f"para prefijo {prefix} página {field_page_number}"
                                )
                            
                except Exception as field_error:
                    self.logger.error(f"Error procesando campos: {field_error}")
                    result['error'] = f"Error en procesamiento de campos: {str(field_error)}"
            
            # Finalizar procesamiento
            processing_time = time.time() - processing_start
            result['processing_time'] = processing_time
            
            return result
                        
        except Exception as e:
            self.logger.error(f"Error procesando imagen {image_path}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'path': str(image_path),
                'processing_time': time.time() - processing_start
            }
            
    def process_fields(self, image: np.ndarray, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for field in fields:
            try:
                # Ajustar x e y para que sean la esquina superior izquierda
                x_top_left = field['x'] - field['ancho'] / 2
                y_top_left = field['y'] - field['alto'] / 2
                # Crear región
                region = RegionCoordinates(
                    x=x_top_left,
                    y=y_top_left,
                    width=field['ancho'],
                    height=field['alto']
                )
                
                # Procesar región según tipo
                result = None
                if field['tipo_campo'] == 'OMR':
                    result = self.process_omr(image, region)
                elif field['tipo_campo'] == 'XMARK':
                    result = self.process_xmark(image, region)
                    
                # Agregar resultado
                if result and result.value is not None:
                    results.append({
                        'campo_id': field['campo_id'],
                        'tipo': field['tipo_campo'],
                        'valor': result.value,
                        'certeza': result.confidence,
                        'metadata': {
                            'pagina': field['pagina'],
                            'indice': field['indice']
                        }
                    })
                        
            except Exception as e:
                self.logger.error(f"Error procesando campo {field.get('campo_id')}: {e}")
                    
        return results
            
    def _get_reference_template(self, path: Path) -> Optional[np.ndarray]:
        """Obtiene o carga una plantilla de referencia."""
        path_str = str(path)
        if path_str not in self._reference_templates:
            try:
                template = cv2.imread(path_str)
                if template is not None:
                    self._reference_templates[path_str] = template
            except Exception as e:
                self.logger.error(f"Error cargando plantilla {path}: {e}")
                return None
                
        return self._reference_templates.get(path_str)

    def read_barcode(self, image: np.ndarray, barcode_regions: Optional[List[Dict[str, float]]]) -> Optional[str]:
        """
        Lee código de barras de tipo Code39 de la imagen.
        
        Args:
            image: Imagen completa
            barcode_regions: Lista de regiones donde buscar códigos
            
        Returns:
            Optional[str]: Código de barras detectado o None
        """
        if image is None:
            return None
            
        try:
            if barcode_regions:
                for region in barcode_regions:
                    # Convertir coordenadas
                    region_coords = RegionCoordinates(
                        x=region['x'],
                        y=region['y'],
                        width=region['width'],
                        height=region['height']
                    )
                    
                    # Extraer región
                    x1, y1, x2, y2 = region_coords.to_pixels(self.dpi, image.shape[:2])
                    roi = image[y1:y2, x1:x2]
                    
                    if roi.size == 0:
                        continue
                    
                    try:
                        # Leer todos los códigos de la región
                        results = zxingcpp.read_barcodes(roi)
                        for result in results:
                            # Verificar si es válido y de tipo Code39
                            if result.valid and result.format == zxingcpp.BarcodeFormat.Code39:
                                self.logger.info(f"Barcode Code39 detectado: {result.text}")
                                return result.text
                            
                    except Exception as e:
                        self.logger.debug(f"Error en lectura de región: {e}")
                        continue
                        
                return None
            
            else:
                try:
                    # Leer todos los códigos de la imagen completa
                    results = zxingcpp.read_barcodes(image)
                    for result in results:
                        # Verificar si es válido y de tipo Code39
                        if result.valid and result.format == zxingcpp.BarcodeFormat.Code39:
                            self.logger.info(f"Barcode Code39 detectado: {result.text}")
                            return result.text
                            
                except Exception as e:
                    self.logger.debug(f"Error en lectura de imagen completa: {e}")
                    
                return None
                
        except Exception as e:
            self.logger.error(f"Error en lectura de códigos: {e}")
            return None
            
        
    def process_omr(self, image: np.ndarray, region: RegionCoordinates) -> Optional[ProcessingResult]:
        """
        Procesa una región OMR y solo devuelve un resultado si la intensidad media es menor o igual a 240.
        """
        try:
            if image is None or not isinstance(image, np.ndarray):
                return None  # Retornamos None si la imagen no es válida

            # Extraer ROI primero
            x1, y1, x2, y2 = region.to_pixels(self.dpi, image.shape[:2])
            if x1 >= x2 or y1 >= y2:
                return None
            
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            # Procesar solo el ROI
            if len(roi.shape) == 3:
                roi_sin_rosa = self.remove_pink_marks(roi)
                roi_gray = cv2.cvtColor(roi_sin_rosa, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi.copy()

            # Calcular intensidad media
            mean_intensity = np.mean(roi_gray)

            # Verificar si la intensidad media es menor o igual a 240
            if mean_intensity <= 240:
                return ProcessingResult(
                    value=mean_intensity,
                    confidence=mean_intensity,
                    metadata={
                        'roi_shape': roi_gray.shape,
                        'mean_intensity': mean_intensity
                    }
                )
            else:
                # Si la intensidad es mayor a 240, no retornamos ningún resultado
                return None

        except Exception as e:
            self.logger.error(f"Error en procesamiento OMR: {e}", exc_info=True)
            return None

        
    def process_xmark(self, image: np.ndarray, region: RegionCoordinates) -> Optional[ProcessingResult]:
        """
        Procesa una región XMARK. Solo retorna resultados si el porcentaje marcado es mayor a 2%.
        """
        try:
            if image is None or not isinstance(image, np.ndarray):
                return None

            # Extraer ROI primero
            x1, y1, x2, y2 = region.to_pixels(self.dpi, image.shape[:2])
            if x1 >= x2 or y1 >= y2:
                return None
            
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            # Procesar solo el ROI
            if len(roi.shape) == 3:
                roi_sin_rosa = self.remove_pink_marks(roi)
                roi_gray = cv2.cvtColor(roi_sin_rosa, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi.copy()
                
            _, roi_threshold = cv2.threshold(roi_gray, 230, 255, cv2.THRESH_BINARY)

            marked_percentage = np.sum(roi_threshold == 0) / roi_threshold.size * 100
            
            # Solo retornar resultado si el porcentaje es mayor a 2
            if marked_percentage > 3:
                return ProcessingResult(
                    value=marked_percentage,
                    confidence=int(marked_percentage),
                    metadata={
                        'roi_shape': roi_threshold.shape,
                        'marked_percentage': marked_percentage
                    }
                )
            return None

        except Exception as e:
            self.logger.error(f"Error en procesamiento XMARK: {e}", exc_info=True)
            return None
        
    # def process_icr(self, image: np.ndarray, region: RegionCoordinates, field: Dict[str, Any], is_numeric: bool) -> ProcessingResult:
    #     try:
    #         num_divisions = field.get('longitud', 1)
            
    #         if not num_divisions or num_divisions < 1:
    #             self.logger.warning(f"Longitud inválida para campo {field.get('campo_id')}, usando 1")
    #             num_divisions = 1
            
    #         region_dict = {
    #             'x': int(region.x * self.dpi),
    #             'y': int(region.y * self.dpi),
    #             'width': int(region.width * self.dpi),
    #             'height': int(region.height * self.dpi)
    #         }
            
    #         #temp_path = Path(self.config.APP_ROOT) / "temp" / "temp_icr.png"
    #         #cv2.imwrite(str(temp_path), image)
            
    #         result = self.icr_processor.process_region(
    #             image_path=str(temp_path),
    #             region=region_dict,
    #             num_divisions=num_divisions,
    #             is_digit=is_numeric  # Usar el parámetro recibido
    #         )
            
    #         return ProcessingResult(
    #             value=result.value,
    #             confidence=result.confidence,
    #             metadata={
    #                 'type': 'ICRN' if is_numeric else 'ICRL',
    #                 **result.metadata
    #             }
    #         )
                
    #     except Exception as e:
    #         self.logger.error(f"Error en procesamiento ICR: {e}")
    #         return ProcessingResult(value=None, confidence=0.0)
        
        
    def process_region(self, 
                      image: np.ndarray,
                      region: RegionCoordinates,
                      field_type: str,
                      columns: int) -> ProcessingResult:
        """
        Procesa una región específica según su tipo.
        
        Args:
            image: Imagen alineada
            region: Coordenadas de la región
            field_type: Tipo de campo (OMR, ICR, XMARK)
            
        Returns:
            ProcessingResult con el resultado
        """
        if field_type == 'OMR':
            return self.process_omr(image, region)
        elif field_type == 'XMARK':
            return self.process_xmark(image, region)
        elif field_type == 'ICR':
            return self.process_icr(image, region, columns)
        else:
            self.logger.warning(f"Tipo de campo no implementado: {field_type}")
            return ProcessingResult(
                value=None,
                confidence=0.0,
                metadata={'status': 'not_implemented'}
            )

    def get_page_number(self, barcode: Optional[str]) -> Optional[int]:
        """
        Determina el número de página basado en el código de barras.
        
        Args:
            barcode: Código de barras o None
            
        Returns:
            Optional[int]: Número de página o None
        """
        if not barcode:
            return None
            
        try:
            # Los últimos dos dígitos indican la página
            page = int(barcode[-2:])
            return page
        except (ValueError, IndexError):
            return None
            
    def get_prefix(self, barcode: Optional[str]) -> Optional[str]:
        """
        Extrae el prefijo del código de barras.
        
        Args:
            barcode: Código de barras o None
            
        Returns:
            Optional[str]: Prefijo o None
        """
        if not barcode:
            return None
            
        try:
            # Los primeros tres dígitos son el prefijo
            return barcode[:3]
        except IndexError:
            return None

    def is_first_page(self, barcode: Optional[str]) -> bool:
        """
        Determina si es la primera página de un examen.
        
        Args:
            barcode: Código de barras o None
            
        Returns:
            bool: True si es primera página (termina en "01" para códigos de 12 dígitos,
            o es el primer código encontrado para otros formatos)
        """
        if not barcode:
            return False
            
        try:
            # Si es un código de 12 dígitos, verificar que termine en "01"
            if len(barcode) == 12:
                return barcode[-2:] == "01"
            # Para otros formatos, considerar como primera página
            return True
        except IndexError:
            return False

    @staticmethod
    def _to_pil(image: np.ndarray):
        """Convierte imagen OpenCV a PIL."""
        from PIL import Image
        return Image.fromarray(image)
    
    def validate_barcode(self, barcode: str) -> bool:
        """
        Valida el formato del código de barras.
        
        Args:
            barcode: Código a validar
            
        Returns:
            bool: True si es válido
        """
        if not barcode or not isinstance(barcode, str):
            return False
            
        # Validar que sea numérico
        if not barcode.isdigit():
            return False
            
        # Para códigos de 12 dígitos, aplicar validación estricta
        if len(barcode) == 12:
            page = int(barcode[-2:])
            if not (1 <= page <= 99):
                return False
        # Para otros formatos, solo validar que sea numérico (ya validado arriba)
        else:
            self.logger.info(f"Código de barras no estándar detectado: {barcode}")
            
        return True

    def remove_pink_marks(self, image: np.ndarray) -> np.ndarray:
        """Elimina marcas en color rosa/magenta."""
        try:
            if len(image.shape) != 3:
                return image
                
            # Convertir a HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            lower_magenta = np.array([130, 8, 180])     # Saturación mínima 8, brillo mínimo 70
            upper_magenta = np.array([180, 255, 255])  

            lower_red = np.array([0, 8, 180])           # Saturación mínima 8, brillo mínimo 70
            upper_red = np.array([50, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_magenta, upper_magenta)
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Crear máscara
            #mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_not(mask)
            
            # Aplicar máscara
            result = cv2.bitwise_and(image, image, mask=mask)
            
            # Rellenar área con blanco
            white = np.ones_like(image) * 255
            result = cv2.add(
                result,
                cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))
            )
            
            return result
            
        except Exception as e:
            print(f"Error removiendo marcas: {e}")
            return image

        