import logging
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import threading
import queue
from datetime import datetime

from core.config import AppConfig
from core.exceptions import ValidationError
from database.service import DatabaseService
from processing.image_processor import ImageProcessor

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BatchInfo:
    ruta: Path
    total_imagenes: int
    estado: str = "pendiente"
    progreso: float = 0.0
    
    # Campos extra para manejar los datos de la tabla
    caja: Optional[str] = None
    pallet: Optional[str] = None
    imagenes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def identificador(self) -> str:
        # Si quieres que muestre algo más amigable:
        if self.caja:
            return f"Caja {self.caja} (Total: {self.total_imagenes})"
        else:
            return str(self.ruta)  # o lo que prefieras
        
class BatchProcessor:
    """Procesador de lotes con recorrido recursivo de directorios."""
    
    def __init__(self, root_path: Optional[Path] = None):
        self.root_path: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
        
        if root_path:
            self.set_root_path(root_path)
            
    def set_root_path(self, path: Path) -> None:
        """Establece y valida la ruta raíz."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Ruta no existe: {path}")
        if not path.is_dir():
            raise ValueError(f"Ruta no es un directorio: {path}")
            
        self.root_path = path
        self.logger.info(f"Ruta raíz establecida: {path}")

    def get_all_batch_info(self) -> List[BatchInfo]:
        """Obtiene información de todos los lotes en la estructura."""
        self._check_root_path()
        batch_infos = []

        for dirpath, dirnames, filenames in os.walk(self.root_path):
            path = Path(dirpath)
            image_files = self._get_image_paths(path)
            if image_files:
                batch_infos.append(BatchInfo(
                    ruta=path,
                    total_imagenes=len(image_files)
                ))
                                   
        return batch_infos

    def _check_root_path(self) -> None:
        """Verifica que exista una ruta raíz."""
        if not self.root_path:
            raise ValueError("Ruta raíz no establecida")
    
    def set_imagenes_db(self, lista_imagenes: List[Dict[str, Any]]) -> None:
        """
        Ajusta el procesador para que trabaje con un listado de rutas/imágenes 
        que provienen de la BD (DIGITALIZACION.dbo.ImagenesProcesadas) en lugar de un root_path.
        """
        self.db_image_list = lista_imagenes[:]  # Copia local
        self.logger.info(f"Se han establecido {len(lista_imagenes)} imágenes de la BD para procesar.")

    def get_all_batch_info_db(self) -> List[BatchInfo]:
        """
        Retorna la información de 'lotes' simulando lo que hace get_all_batch_info,
        pero usando self.db_image_list (si deseas agruparlas en un solo lote).
        """
        if not hasattr(self, 'db_image_list'):
            raise ValueError("No se ha establecido la lista de imágenes de BD (llamar set_imagenes_db primero).")

        # Aquí decides si haces un solo BatchInfo o varios. 
        # Por ahora, retornamos un único BatchInfo con todas las imágenes:
        batch_info_list = []
        
        if self.db_image_list:
            # Ejemplo: un solo "lote"
            bi = BatchInfo(
                ruta=Path("[BD]"),  # algo simbólico, ya que no hay carpeta real
                total_imagenes=len(self.db_image_list),
                imagenes=self.db_image_list  # la lista completa
            )
            batch_info_list.append(bi)
        
        return batch_info_list
            
    @staticmethod
    def _get_image_paths(path: Path) -> List[Path]:
        """Obtiene las rutas de imágenes válidas ordenadas (no recursivo)."""
        def extract_number(filename: str) -> int:
            """Extrae número de un nombre de archivo."""
            parts = filename.split('-')
            try:
                return int(parts[-1].split('.')[0])
            except (ValueError, IndexError):
                return 0

        image_files = [
            f for f in path.glob('*') 
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        ]
        return sorted(image_files, key=lambda x: extract_number(x.stem))

class BatchProcessingThread(threading.Thread):
    """Thread para procesamiento de lotes."""
    
    def __init__(self,
                 db_service: DatabaseService,
                 config: AppConfig,
                 batch_info: BatchInfo):
        """Inicializa el thread de procesamiento."""
        super().__init__()
        self.daemon = True  # Permitir que el hilo principal termine sin esperar este hilo
        
        # Inicializar logger
        self.logger = logging.getLogger(__name__)
        
        # En lugar de señales, usa callbacks
        self.on_result = None
        self.on_error = None
        self.on_batch_completed = None
        
        # El resto se mantiene igual...
        self.db_service = db_service
        self.config = config
        self.batch_info = batch_info
        self.image_processor = ImageProcessor(config)
        
        # Tracking mejorado de exámenes
        self.current_exam = None
        self.last_valid_exam = None
        self.current_references = None
        
        # Control de estado
        self.should_stop = False
        self.is_paused = False
        self._pause_condition = threading.Condition()
        
        # Inicializar image_paths
        self.image_paths = []
        for img_data in batch_info.imagenes:
            ruta_str = img_data.get('ruta', '')
            if ruta_str:
                self.image_paths.append(Path(ruta_str))
        
        # Tracking de numeración de imágenes
        self.current_image_number = 1  # Número secuencial de imagen
        self.current_exam_code = None  # Para tracking del examen actual (10 primeros dígitos)
        self.last_reference_pages = False
                    
        # Estadísticas
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'images_no_barcode': 0,
            'total_aligned': 0,
            'start_time': None
        }

        self.existing_barcodes = set()
        self.barcodes_initialized = False

        try:
            self.existing_barcodes = self.db_service.repository.get_all_barcodes()
            self.logger.info(f"Cargados {len(self.existing_barcodes)} códigos de barra existentes")
        except Exception as e:
            self.logger.error(f"Error cargando códigos de barra existentes: {e}")

    def run(self):
        try:
            self.stats = {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'images_no_barcode': 0,
                'total_aligned': 0,
                'start_time': None
            }

            self.logger.info(f"Iniciando procesamiento de lote: {self.batch_info.identificador}")
            self.stats['start_time'] = time.time()
            
            # Método de callback en lugar de signal
            # En lugar de: self.batch_started.emit(self.batch_info)
            # Simplemente no hacemos nada, o logeamos
            
            if not self.image_paths:
                error_msg = "No se encontraron imágenes para procesar en este lote (self.image_paths está vacío)"
                if self.on_error:
                    self.on_error(error_msg)
                raise ValidationError(error_msg)

            total_images = len(self.image_paths)

            for i, image_path in enumerate(self.image_paths, 1):
                if self.should_stop:
                    break

                with self._pause_condition:
                    while self.is_paused and not self.should_stop:
                        self._pause_condition.wait()

                try:
                    # Procesar imagen individual
                    result = self.process_single_image(image_path)
                    
                    # Usar callback en lugar de señal
                    if self.on_result:
                        self.on_result(result)

                    # Actualizar estadísticas
                    self.stats['processed'] += 1
                    if result['status'] == 'success':
                        self.stats['successful'] += 1
                        if result.get('aligned', False):
                            self.stats['total_aligned'] += 1
                    else:
                        self.stats['failed'] += 1

                    # Actualizar progreso
                    progress = (i / total_images) * 100
                    # En lugar de: self.progress.emit(progress)
                    # Simplemente actualizamos el valor interno
                    self.batch_info.progreso = progress

                except Exception as e:
                    self.logger.error(f"Error procesando imagen {image_path}: {e}")
                    if self.on_error:
                        self.on_error(str(e))
                    self.stats['failed'] += 1

            # Finalizar último examen si existe
            if self.current_exam:
                self._finalize_current_exam()

            # Usar callback en lugar de emit
            if self.on_batch_completed:
                self.on_batch_completed(self._get_final_stats())

        except Exception as e:
            self.logger.error(f"Error en procesamiento de lote: {e}")
            if self.on_error:
                self.on_error(str(e))


    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Procesa una imagen individual según la lógica de negocio,
        pero usando 'barcode_c39' proveniente de la tabla
        en lugar de detectar barcode desde la imagen.
        """
        try:
            # 1) Buscar la info en batch_info.imagenes que coincida con 'image_path'
            barcode_c39 = None
            for img_data in self.batch_info.imagenes:
                # Comparar la ruta. Si coincide, usamos su 'barcode_c39'
                if Path(img_data['ruta']) == image_path:
                    barcode_c39 = img_data.get('barcode_c39')
                    break

            # 2) Construimos un 'result' estilo el que generaba 'image_processor'
            #    pero metemos 'barcode' con el valor de 'barcode_c39'.
            #    También ponemos 'status' en 'success' si hay barcode,
            #    o 'success' con logic condicional; ajusta si quieres error si no hay.
            if barcode_c39:
                result = {
                    'status': 'success',
                    'barcode': barcode_c39,
                    'path': str(image_path),
                    'prefix': None,           # Lo calcularás abajo,
                    'aligned': False,
                    'processing_time': 0,
                }
            else:
                # Si no hay barcode en la tabla, creamos un result sin 'barcode'
                result = {
                    'status': 'success',      # O 'warning', como quieras
                    'barcode': None,
                    'path': str(image_path),
                    'prefix': None,
                    'aligned': False,
                    'processing_time': 0,
                }

            # 3) Si hay un 'barcode_c39', podemos extraer el 'prefix'
            #    por ejemplo, si tu prefix son los 3 primeros dígitos:
            if barcode_c39 and len(barcode_c39) >= 3:
                result['prefix'] = barcode_c39[:3]

            # 4) Con este 'result', reusamos la misma lógica:
            #    - si hay barcode -> _handle_image_with_barcode
            #    - si no hay -> _handle_image_without_barcode
            if result.get('barcode'):
                return self._handle_image_with_barcode(image_path, result)
            else:
                return self._handle_image_without_barcode(image_path, result)

        except Exception as e:
            self.logger.error(f"Error procesando imagen {image_path}: {e}")
            return self._create_error_result(image_path, str(e))

    def _ensure_barcode_exists(self, barcode: str) -> bool:
        """
        Verifica si un código de barras ya existe en la base de datos.
        Optimizado para evitar cargar todos los códigos al inicio.
        """
        if not barcode:
            return False
            
        # Si ya está en el conjunto local, ya existe
        if barcode in self.existing_barcodes:
            return True
            
        # Verificar directamente en la base de datos
        examen_existente = self.db_service.buscar_examen_por_codigo(barcode)
        
        # Si existe, agregar al conjunto local para futuras consultas
        if examen_existente:
            self.existing_barcodes.add(barcode)
            return True
            
        return False      
    
    def _handle_image_with_barcode(self, image_path: Path, initial_result: Dict) -> Dict[str, Any]:
        """Maneja el procesamiento de una imagen con código de barras."""
        try:
            barcode = initial_result['barcode']
            prefix = initial_result.get('prefix')
            if len(barcode) == 12:
                exam_code = barcode[:-2]  # Primeros 10 dígitos identifican el examen
            else: 
                exam_code = barcode

            if not prefix:
                raise ValidationError(f"Código de barras sin prefijo válido: {barcode}")

            # Si es un nuevo examen (código diferente), reiniciar numeración
            if exam_code != self.current_exam_code:
                self.current_image_number = 1
                self.last_reference_pages = False
                self.current_exam_code = exam_code
                
                # Si había un examen previo, finalizarlo
                if self.current_exam:
                    self._finalize_current_exam()
                
                # Crear nuevo examen
                exam_id = self._create_new_exam(barcode)
                
                # Obtener referencias validadas para el nuevo examen
                self.current_references = self._get_validated_references(prefix)
                
                # Configurar nuevo examen
                plantilla = self.db_service.get_plantilla_by_prefix(prefix)
                self.current_exam = {
                    'exam_id': exam_id,
                    'barcode': barcode,
                    'prefix': prefix,
                    'current_page': int(barcode[-2:]),  # Página física del examen
                    'start_time': time.time(),
                    'paginas_variables': plantilla.get('paginas_variables', False),
                    'nro_paginas': plantilla.get('nro_paginas', 0),
                    'last_reference_index': plantilla.get('nro_paginas', 0) - 2 if plantilla.get('paginas_variables', False) else plantilla.get('nro_paginas', 0)
                }

            # Procesar y registrar imagen con la referencia correspondiente
            result = self._process_image_with_reference(image_path)
            
            # Actualizar contadores para siguiente imagen
            self.current_image_number += 1
            self.last_valid_exam = self.current_exam.copy()
            
            return result

        except Exception as e:
            self.logger.error(f"Error procesando imagen con código: {e}")
            return self._create_error_result(image_path, str(e))

    def _handle_image_without_barcode(self, image_path: Path, initial_result: Dict) -> Dict[str, Any]:
        """Maneja el procesamiento de una imagen sin código de barras."""
        try:
            if not self.last_valid_exam or not self.current_exam_code:
                return self._create_error_result(
                    image_path,
                    "Imagen sin código encontrada sin examen previo válido"
                )

            # Usar el último examen válido como referencia
            self.current_exam = self.last_valid_exam.copy()
            
            # Procesar y registrar imagen
            result = self._process_image_with_reference(image_path)
            
            # Actualizar contadores para siguiente imagen
            self.current_image_number += 1
            self.last_valid_exam = self.current_exam.copy()
            
            return result

        except Exception as e:
            self.logger.error(f"Error procesando imagen sin código: {e}")
            return self._create_error_result(image_path, str(e))
    
    
    def _process_image_with_reference(self, image_path: Path) -> Dict[str, Any]:
        """Procesa una imagen con la referencia correspondiente."""
        try:
            # Determinar índice de referencia a usar
            reference_index = self._get_reference_index()
            
            self.logger.debug(
                f"Procesando imagen {image_path.name} - "
                f"Examen: {self.current_exam_code}, "
                f"Número: {self.current_image_number}, "
                f"Usando referencia: {reference_index + 1}"
            )

            # Procesar imagen
            result = self.image_processor.process_image(
                image_path=image_path,
                barcode_regions= None,
                reference_paths=self.current_references,
                db_service=self.db_service,
                reference_index=reference_index,  # Pasamos explícitamente el índice
                prefix=self.current_exam['prefix']  # Agregar esta línea
            )

            if result['status'] == 'success':
                image_id = self.db_service.registrar_imagen(
                    exam_id=self.current_exam['exam_id'],
                    ruta=str(image_path),
                    pagina=self.current_image_number,
                    alineada=result.get('aligned', False)
                )

                if result.get('results'):
                    self.db_service.guardar_resultados([{
                        'imagen_id': image_id,
                        **r
                    } for r in result['results']])

            return result

        except Exception as e:
            self.logger.error(f"Error en _process_image_with_reference: {e}")
            raise

    def _get_validated_references(self, prefix: str) -> List[Path]:
        """Obtiene y valida las referencias para un prefijo."""
        references = self._get_reference_paths(prefix)
        
        # Validar secuencia de referencias
        plantilla = self.db_service.get_plantilla_by_prefix(prefix)
        expected_pages = plantilla.get('nro_paginas', 0)
        
        if not expected_pages:
            raise ValidationError(
                f"Plantilla {prefix} no tiene número de páginas definido"
            )
            
        if len(references) != expected_pages:
            raise ValidationError(
                f"Número incorrecto de referencias para prefijo {prefix}. "
                f"Esperadas: {expected_pages}, Encontradas: {len(references)}"
            )
            
        return references
    
    def _finalize_current_exam(self):
        """Finaliza el examen actual y actualiza su estado."""
        if not self.current_exam:
            return
            
        try:
            self.db_service.actualizar_estado_examen(
                self.current_exam['exam_id'],
                estado='1'  # Procesado
            )
            
            self.logger.info(
                f"Finalizado examen {self.current_exam['barcode']} "
            )
            
        except Exception as e:
            self.logger.error(f"Error finalizando examen: {e}")
            raise
        
    def _create_error_result(self, image_path: Path, error_msg: str) -> Dict[str, Any]:
        """Crea un resultado de error estandarizado."""
        return {
            'path': str(image_path),
            'status': 'error',
            'barcode': None,
            'page': None,
            'exam_code': None,
            'aligned': False,
            'processing_time': 0,
            'error': error_msg
        }

    def _create_new_exam(self, barcode: str) -> int:
        """Crea un nuevo examen en la base de datos."""
        try:
            # Validar que el barcode tiene formato adecuado
            if len(barcode) == 12 and barcode.endswith('01'):
                self.logger.debug(f"Creando nuevo examen con barcode de 12 dígitos: {barcode}")
            elif len(barcode) == 10:
                self.logger.debug(f"Creando nuevo examen con barcode de 10 dígitos: {barcode}")
            elif len(barcode) == 9:
                self.logger.debug(f"Creando nuevo examen con barcode de 9 dígitos: {barcode}")
            else:
                raise ValidationError("El barcode no cumple con los requisitos para crear un nuevo examen")

            # Buscar si ya existe - usando la función optimizada
            if self._ensure_barcode_exists(barcode):
                # Obtener examen existente
                examen_existente = self.db_service.buscar_examen_por_codigo(barcode)
                if examen_existente:
                    return examen_existente['examen_id']
            
            # No existe, crear nuevo examen
            prefix = self.image_processor.get_prefix(barcode)
            if not prefix:
                raise ValidationError("No se pudo obtener prefijo del código de barras")

            # Crear examen
            exam_id = self.db_service.crear_examen(
                codigo_barras=barcode,
                plantilla_id=self._get_plantilla_id(prefix)
            )

            # Añadir al conjunto para futuras referencias
            self.existing_barcodes.add(barcode)
            
            self.logger.info(f"Creado nuevo examen: {exam_id} - {barcode}")
            return exam_id

        except Exception as e:
            self.logger.error(f"Error creando examen: {e}")
            raise
        
    def _get_reference_index(self) -> int:
        """Determina el índice de referencia a usar para la imagen actual."""
        try:
            if not self.current_exam or not self.current_references:
                return 0

            total_referencias = len(self.current_references)
            
            if self.current_exam.get('paginas_variables', False):
                last_fixed_page = self.current_exam.get('last_reference_index', 0)
                
                # Si aún no llegamos a la última página fija
                if self.current_image_number <= last_fixed_page:
                    return self.current_image_number - 1
                else:
                    # Alternar entre las dos últimas referencias en orden correcto (e.g: 7,8,7,8,...)
                    penultima_ref = total_referencias - 2  
                    return penultima_ref + ((self.current_image_number - last_fixed_page - 1) % 2)
            else:
                # Para exámenes fijos, usar la secuencia normal
                return min(self.current_image_number - 1, total_referencias - 1)

        except Exception as e:
            self.logger.error(f"Error calculando índice de referencia: {e}")
            return 0

    def _get_reference_paths(self, prefix: str) -> List[Path]:
        """
        Obtiene las rutas de las imágenes de referencia para un prefijo.
        
        Args:
            prefix: Prefijo del código de barras
                
        Returns:
            List[Path]: Lista de rutas a imágenes de referencia ordenadas por página
        """
        try:
            plantilla = self.db_service.get_plantilla_by_prefix(prefix)
            base_path = Path(plantilla['ruta_imagen'])

            if not base_path.exists() or not base_path.is_dir():
                raise ValidationError(
                    f"Ruta de imágenes de referencia no válida: {plantilla['ruta_imagen']}"
                )

            # Obtener y ordenar imágenes de referencia
            reference_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                reference_files.extend(base_path.glob(f"*{ext}"))
                
            def extract_number(path: Path) -> int:
                """
                Extrae el número del nombre del archivo, ignorando extensión y prefijos.
                Maneja casos como: '1.jpg', '01.jpg', 'prefix_1.jpg', etc.
                """
                try:
                    # Eliminar extensión y dividir por guiones bajos
                    name = path.stem
                    # Extraer todos los números del nombre
                    numbers = ''.join(filter(str.isdigit, name))
                    return int(numbers) if numbers else 0
                except (ValueError, IndexError):
                    return 0
                
            # Ordenar archivos numéricamente
            reference_files.sort(key=extract_number)

            if not reference_files:
                raise ValidationError(
                    f"No se encontraron imágenes de referencia en: {base_path}"
                )

            self.logger.info(
                f"Encontradas {len(reference_files)} imágenes de referencia "
                f"para prefijo {prefix} en {base_path}"
            )

            return reference_files

        except Exception as e:
            self.logger.error(
                f"Error obteniendo rutas de referencia para prefijo {prefix}: {e}"
            )
            raise ValidationError(f"Error obteniendo referencias: {str(e)}")

    def _get_plantilla_id(self, prefix: str) -> int:
        """
        Obtiene el ID de plantilla para un prefijo.
        
        Args:
            prefix: Prefijo del código de barras (3 primeros dígitos)
                
        Returns:
            int: ID de la plantilla asociada al prefijo
                
        Raises:
            ValidationError: Si no se encuentra plantilla para el prefijo
        """
        try:
            # Obtener plantillas y buscar la que corresponde al prefijo
            plantilla = self.db_service.get_plantilla_by_prefix(prefix)
            
            if plantilla and plantilla["plantilla_id"]:
                return plantilla["plantilla_id"]

            raise ValidationError(f"No se encontró plantilla para el prefijo: {prefix}")
            
        except Exception as e:
            self.logger.error(f"Error obteniendo ID de plantilla para prefijo {prefix}: {e}")
            raise ValidationError(f"Error buscando plantilla: {str(e)}")

    def _emit_final_stats(self):
        """Emite estadísticas finales del procesamiento."""
        try:
            end_time = time.time()
            processing_time = end_time - self.stats['start_time']
            
            self.batch_completed.emit({
                'batch_info': self.batch_info,
                'total_processed': self.stats['processed'],
                'successful': self.stats['successful'],
                'failed': self.stats['failed'],
                'total_aligned': self.stats['total_aligned'],  # Añadir total de alineadas
                'processing_time': processing_time,
                'success_rate': (self.stats['successful'] / self.stats['processed'] * 100)
                    if self.stats['processed'] > 0 else 0,
                'completion_date': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error emitiendo estadísticas: {e}")

    def _get_final_stats(self):
        """Prepara las estadísticas finales para el callback."""
        end_time = time.time()
        processing_time = end_time - self.stats['start_time']
        
        return {
            'batch_info': self.batch_info,
            'total_processed': self.stats['processed'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'total_aligned': self.stats['total_aligned'],
            'processing_time': processing_time,
            'success_rate': (self.stats['successful'] / self.stats['processed'] * 100)
                if self.stats['processed'] > 0 else 0,
            'completion_date': datetime.now().isoformat()
        }

    def pause(self):
        """Pausa el procesamiento."""
        with self._pause_condition:
            self.is_paused = True
            self.logger.info("Procesamiento pausado")

    def resume(self):
        """Reanuda el procesamiento."""
        with self._pause_condition:
            self.is_paused = False
            self._pause_condition.notify_all()
            self.logger.info("Procesamiento reanudado")

    def stop(self):
        """Detiene el procesamiento."""
        with self._pause_condition:
            self.should_stop = True
            self.is_paused = False
            self._pause_condition.notify_all()
            self.logger.info("Procesamiento detenido")