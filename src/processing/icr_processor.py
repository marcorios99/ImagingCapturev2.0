import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageOps
from dataclasses import dataclass
from keras.models import load_model

@dataclass
class ICRResult:
    """Resultado del procesamiento ICR."""
    value: str
    confidence: float
    metadata: Dict[str, Any] = None

class ICRProcessor:
    """Procesador de caracteres ICR que mantiene la lógica exacta de los modelos originales."""
    
    # Mapeo de índices a letras (igual que en LetterRecognizer original)
    index_to_letter = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Ñ'
    }
    
    def __init__(self, config: Any):
        """Inicializa el procesador con configuración."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuración de procesamiento
        self.dpi = getattr(config, 'dpi', 300)
        
        # Inicializar ICR Processor
        models_path = Path(config.APP_ROOT) / 'ml_models'
        self.icr_processor = ICRProcessor(models_path)
        
        # Cache para plantillas de referencia
        self._reference_templates: Dict[str, np.ndarray] = {}

    def remove_pink_color_digit(self, image):
        """Eliminar color rosa para dígitos (umbral específico)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([130, 0, 0])
        upper_pink = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        mask_inv = cv2.bitwise_not(mask)
        filtered_image = cv2.bitwise_and(image, image, mask=mask_inv)
        white_image = np.ones_like(image) * 255
        return cv2.add(filtered_image, cv2.bitwise_and(white_image, white_image, mask=mask))

    def remove_pink_color_letter(self, image):
        """Eliminar color rosa para letras (umbral específico)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 20, 20])
        upper_pink = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        mask_inv = cv2.bitwise_not(mask)
        filtered_image = cv2.bitwise_and(image, image, mask=mask_inv)
        white_image = np.ones_like(image) * 255
        return cv2.add(filtered_image, cv2.bitwise_and(white_image, white_image, mask=mask))

    def preprocess_cropped_image(self, pil_img, is_digit=True):
        """Preprocesamiento exacto como en las clases originales."""
        img_gray = pil_img.convert('L')
        img_gray = ImageOps.autocontrast(img_gray, cutoff=0)
        img_array = np.array(img_gray)
        kernel = np.ones((2, 2), np.uint8)
        img_array = cv2.erode(img_array, kernel, iterations=1)
        img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        # Umbral diferente para dígitos y letras
        threshold = 215 if is_digit else 200
        _, img_array = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
        
        img_array = img_array / 255.0
        img_array = np.array(Image.fromarray((img_array * 255).astype(np.uint8)).resize((28, 28))) / 255.0
        img_array = img_array.reshape(28, 28, 1)
        return img_array

    def gray_scale(self, pil_img, is_digit=True):
        """Conversión a escala de grises con eliminación de rosa específica."""
        image_array = np.array(pil_img)
        image_sin_rosa = self.remove_pink_color_digit(image_array) if is_digit else self.remove_pink_color_letter(image_array)
        return cv2.cvtColor(image_sin_rosa, cv2.COLOR_BGR2GRAY)

    def is_blank(self, img_array, threshold=None, is_digit=True):
        """Verifica si una región está en blanco usando umbrales específicos."""
        if threshold is None:
            threshold = 254 if is_digit else 252
        return np.mean(img_array) > threshold

    def process_region(self, image_path: str, region: Dict[str, float], num_divisions: int, is_digit: bool = True) -> ICRResult:
        """
        Procesa una región ICR usando la misma lógica que las clases originales.
        
        Args:
            image_path: Ruta a la imagen
            region: Diccionario con x, y, width, height
            num_divisions: Número de caracteres esperados
            is_digit: True para usar modelo de dígitos, False para letras
        """
        try:
            # Cargar y preprocesar imagen
            img = cv2.imread(str(image_path))
            preprocessed_image = self.remove_pink_color_digit(img) if is_digit else self.remove_pink_color_letter(img)
            img_np = np.array(preprocessed_image)
            
            # Extraer coordenadas
            x1, y1 = int(region['x']), int(region['y'])
            width, height = int(region['width']), int(region['height'])
            x2, y2 = x1 + width, y1 + height
            
            # Calcular ancho de división
            division_width = width // num_divisions
            
            input_images = []
            blank_indices = []
            predictions = []
            confidences = []
            
            # Procesar cada división
            for j in range(num_divisions):
                x_start = x1 + j * division_width
                x_end = x2 if j == num_divisions - 1 else x_start + division_width
                
                # Extraer y procesar sub-imagen
                sub_img = img_np[y1:y2, x_start:x_end]
                sub_img_pil = Image.fromarray(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))
                gray_image_roi = self.gray_scale(sub_img_pil, is_digit)
                
                if self.is_blank(gray_image_roi, is_digit=is_digit):
                    blank_indices.append(j)
                    predictions.append(' ')
                    continue
                
                preprocessed = self.preprocess_cropped_image(sub_img_pil, is_digit)
                input_images.append(preprocessed)
            
            # Realizar predicciones si hay imágenes no en blanco
            if input_images:
                input_data = np.stack(input_images, axis=0)
                model = self.digit_model if is_digit else self.letter_model
                preds = model.predict(input_data, verbose=0)
                
                pred_indices = np.argmax(preds, axis=1)
                pred_confidences = np.max(preds, axis=1)
                
                # Convertir índices a caracteres
                if is_digit:
                    chars = [str(idx) for idx in pred_indices]
                else:
                    chars = [self.index_to_letter[idx] for idx in pred_indices]
                
                # Reconstruir resultado final
                char_idx = 0
                final_predictions = []
                final_confidences = []
                
                for i in range(num_divisions):
                    if i in blank_indices:
                        final_predictions.append(' ')
                    else:
                        final_predictions.append(chars[char_idx])
                        final_confidences.append(pred_confidences[char_idx])
                        char_idx += 1
                
                value = ''.join(final_predictions)
                confidence = min(final_confidences) * 100 if final_confidences else 0.0
                
                return ICRResult(
                    value=value,
                    confidence=confidence,
                    metadata={
                        'individual_confidences': final_confidences,
                        'is_digit': is_digit
                    }
                )
            
            return ICRResult(value='', confidence=0.0)
            
        except Exception as e:
            self.logger.error(f"Error procesando región ICR: {e}")
            return ICRResult(value='', confidence=0.0)