import logging
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import gc
from collections import defaultdict
from typing import Dict, List, Any
from core.config import AppConfig
from core.exceptions import ValidationError
from database.service import DatabaseService
from processing.batch_processor import BatchProcessor, BatchInfo, BatchProcessingThread
# otras importaciones…
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)



class CLIController:
    """Controlador CLI para el procesamiento de imágenes."""
    
    def __init__(
        self,
        db_service: DatabaseService,
        config: AppConfig
    ):
        """Inicializa el controlador."""
        # Servicios y configuración
        self.db_service = db_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.batch_processor = BatchProcessor()
        
        # Estado de la aplicación
        self.current_state = 'ready'
        self.process_thread = None
        self.all_batches = []
        self.current_batch_index = -1
        self.start_time = None
        self.is_processing = False
        self.is_canceling_all = False
        
        # Cola para resultados
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        
    def run(self, root: Path = None, workers: int = 1):
        try:
            # 1️⃣  Si ya tenemos lotes (vienen de la BD), saltamos la solicitud de carpeta
            if self.all_batches:
                self.process_all_batches()
                return

            # 2️⃣  Comportamiento anterior cuando se trabaja con carpetas
            if root:
                self.select_root_folder(root)
            else:
                root_str = input("Ingrese la ruta de la carpeta raíz: ")
                self.select_root_folder(Path(root_str))

            self.process_all_batches()
            
        except KeyboardInterrupt:
            self.logger.info("Procesamiento interrumpido por el usuario")
            self.stop_processing()
        except Exception as e:
            self.logger.error(f"Error en ejecución: {e}", exc_info=True)
            
    def select_root_folder(self, folder: Path):
        """Selecciona la carpeta raíz y carga la estructura de lotes."""
        try:
            if not folder.exists():
                raise ValidationError(f"La carpeta no existe: {folder}")
                
            # Configurar procesador de lotes
            self.batch_processor.set_root_path(folder)
            self.all_batches = self.batch_processor.get_all_batch_info()
            
            if not self.all_batches:
                raise ValidationError("No se encontraron lotes para procesar")
                
            # Iniciar con el primer lote
            self.current_batch_index = 0
            
            # Mostrar información de lotes encontrados
            self.logger.info(f"Cargados {len(self.all_batches)} lotes de {folder}")
            for i, batch in enumerate(self.all_batches):
                self.logger.info(f"Lote {i+1}: {batch.identificador} - {batch.total_imagenes} imágenes")
            
        except Exception as e:
            self.logger.error(f"Error cargando estructura: {e}", exc_info=True)
            raise ValidationError(f"Error al cargar estructura: {str(e)}")
            
    def process_all_batches(self):
        """Procesa todos los lotes secuencialmente mostrando una barra de progreso."""
        if not self.all_batches:
            self.logger.error("No hay lotes para procesar")
            return

        total_batches = len(self.all_batches)
        self.logger.info(f"Iniciando procesamiento de {total_batches} lotes")

        processed   = 0
        successful  = 0
        failed      = 0
        self.start_time = time.time()

        # Barra de progreso global
        with Progress(
            SpinnerColumn(style="bold magenta"),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=logging.getLogger().handlers[1].console   # usa el mismo Rich console
        ) as progress:

            batch_task = progress.add_task("Lotes", total=total_batches)

            # ─── Recorrido de cada lote ────────────────────────────────
            for i, batch in enumerate(self.all_batches, start=1):
                if self.stop_event.is_set():
                    break

                self.current_batch_index = i - 1
                progress.update(
                    batch_task,
                    description=f"[cyan]Lote {i}/{total_batches}: {batch.identificador}"
                )

                try:
                    result = self.process_batch(batch)
                    processed += 1

                    if result.get("status") == "success":
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    self.logger.error(
                        f"Error procesando lote {batch.identificador}: {e}",
                        exc_info=True
                    )
                    failed += 1

                # Avanzar barra global
                progress.advance(batch_task)

            # Fin del with → Rich refresca y deja el último estado pintado

        # ─── Resumen final (fuera del Progress) ────────────────────────
        total_time = time.time() - self.start_time
        self.logger.info(
            f"Procesamiento completado en {self._format_time(total_time)} - "
            f"Procesados: {processed}/{total_batches} - "
            f"Exitosos: {successful}, Fallidos: {failed}"
        )
        
    def process_batch(self, batch_info: BatchInfo) -> Dict[str, Any]:
        """Procesa un lote individual, optimizando el uso de memoria."""
        try:
            # Inicializar estadísticas
            stats = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'aligned': 0,
                'start_time': time.time()
            }
            
            # Agrupar imágenes por código de examen
            exam_groups = self._group_images_by_exam(batch_info)
            
            total_exams = len(exam_groups)
            total_images = batch_info.total_imagenes
            
            self.logger.info(f"Procesando {total_exams} exámenes con {total_images} imágenes en total")
            
            # Procesar un examen a la vez
            for i, (exam_code, images) in enumerate(exam_groups.items(), 1):
                if self.stop_event.is_set():
                    break
                    
                self.logger.info(f"Procesando examen {i}/{total_exams}: {exam_code} ({len(images)} imágenes)")
                
                # Crear instancia de procesamiento para este examen específico
                exam_processor = BatchProcessingThread(
                    db_service=self.db_service,
                    config=self.config,
                    batch_info=BatchInfo(
                        ruta=batch_info.ruta,
                        total_imagenes=len(images),
                        estado=batch_info.estado,
                        progreso=0.0,
                        caja=batch_info.caja,
                        pallet=batch_info.pallet,
                        imagenes=images  # Solo las imágenes de este examen
                    )
                )
                
                # Procesar todas las imágenes de este examen
                exam_stats = self._process_exam(exam_processor, images)
                
                # Actualizar estadísticas globales
                stats['total_processed'] += exam_stats['total']
                stats['successful'] += exam_stats['successful'] 
                stats['failed'] += exam_stats['failed']
                stats['aligned'] += exam_stats['aligned']
                
                # Reportar progreso
                progress = (i / total_exams) * 100
                elapsed = time.time() - stats['start_time']
                
                self.logger.info(
                    f"Progreso: {progress:.1f}% - {stats['total_processed']}/{total_images} imágenes - "
                    f"Tiempo: {self._format_time(elapsed)} - "
                    f"Exitosas: {stats['successful']}, Fallidas: {stats['failed']}"
                )
                
                # Liberar memoria explícitamente
                del exam_processor
                del images
                gc.collect()
            
            # Resultado final
            processing_time = time.time() - stats['start_time']
            stats['processing_time'] = processing_time
            
            self.logger.info(
                f"Procesamiento completado en {self._format_time(processing_time)} - "
                f"Procesadas: {stats['total_processed']}/{total_images} - "
                f"Exitosas: {stats['successful']}, Fallidas: {stats['failed']}"
            )
            
            return {
                'status': 'success',
                'batch_info': batch_info,
                **stats
            }
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento de lote: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'batch_info': batch_info
            }
        finally:
            gc.collect()
    
    def _setup_thread_callbacks(self):
        """Configura callbacks para el thread de procesamiento."""
        if not self.process_thread:
            return
        
        # Reemplazar señales por funciones de callback
        self.process_thread.on_result = self._on_result_ready
        self.process_thread.on_error = self._on_processing_error
        self.process_thread.on_batch_completed = self._on_processing_complete
    
    def _on_result_ready(self, result):
        """Callback cuando un resultado está listo."""
        self.result_queue.put(result)
        
        # Mostrar información mínima por consola
        status = result.get('status', 'unknown')
        path = result.get('path', 'unknown')
        barcode = result.get('barcode', 'N/A')
        
        self.logger.debug(f"Procesada: {path} - Status: {status} - Barcode: {barcode}")
    
    def _on_processing_error(self, error_msg):
        """Callback cuando hay un error en el procesamiento."""
        self.logger.error(f"Error en procesamiento: {error_msg}")
    
    def _on_processing_complete(self, stats):
        """Callback cuando se completa el procesamiento de un lote."""
        self.logger.info(
            f"Lote completado - "
            f"Procesadas: {stats['total_processed']} - "
            f"Exitosas: {stats['successful']}, Fallidas: {stats['failed']}"
        )
    
    def stop_processing(self):
        """Detiene el procesamiento actual."""
        self.logger.info("Deteniendo procesamiento...")
        
        self.stop_event.set()
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.stop()  # Debes adaptar este método en BatchProcessingThread
            self.process_thread.join(timeout=5)
            
            if self.process_thread.is_alive():
                self.logger.warning("El hilo de procesamiento no se detuvo correctamente")
        
        self.logger.info("Procesamiento detenido")
    
    def _format_time(self, seconds: float) -> str:
        """Formatea un tiempo en segundos a formato HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def filtrar_por_codigos_examen(self, exam_codes: List[str]):
        """Filtra resultados por códigos de examen."""
        try:
            # Obtener registros filtrados
            imagenes_db = self.db_service.get_imagenes_digitalizacion(exam_codes=exam_codes)
            if not imagenes_db:
                raise ValidationError("No se encontraron imágenes para los CódigosExamen especificados.")
            
            # Agrupar por 'codigo_examen'
            from collections import defaultdict
            lotes_por_examen = defaultdict(list)
            
            for img in imagenes_db:
                cod_ex = img.get('codigo_examen', 'SIN_EXAM')
                lotes_por_examen[cod_ex].append(img)
            
            # Crear self.all_batches con 1 BatchInfo por examen
            self.all_batches = []
            for cod_ex, imgs in lotes_por_examen.items():
                batch_info = BatchInfo(
                    ruta=Path(f"[DB]/Examen_{cod_ex}"),  # simbólico, no es un folder real
                    total_imagenes=len(imgs),
                    estado="pendiente",
                    progreso=0.0,
                    caja=None,        
                    pallet=None,      
                    imagenes=imgs     # TODAS las imágenes de este examen
                )
                self.all_batches.append(batch_info)
            
            if not self.all_batches:
                raise ValidationError("No se crearon lotes con los CódigosExamen indicados.")
            
            # Iniciar con el primer lote
            self.current_batch_index = 0
            
            self.logger.info(
                f"Se cargaron {len(self.all_batches)} exámenes (lotes) desde la base de datos."
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al filtrar por CódigosExamen: {e}", exc_info=True)
            return False
    # ------------------------------------------------------------------
    # 1) Agrupa las imágenes de un BatchInfo según el código de examen
    # ------------------------------------------------------------------
    def _group_images_by_exam(self, batch_info) -> Dict[str, List[Dict[str, Any]]]:
        """
        Devuelve un dict {codigo_examen: [imagenes…]} usando
        el campo 'codigo_examen' que llega desde la tabla DIGITALIZACION.
        """
        groups = defaultdict(list)
        for img in batch_info.imagenes:
            exam_code = img.get('codigo_examen', 'SIN_EXAM')
            groups[exam_code].append(img)
        return groups

    # ------------------------------------------------------------------
    # 2) Procesa sincrónicamente un examen con BatchProcessingThread
    # ------------------------------------------------------------------
    def _process_exam(self, exam_processor, images) -> Dict[str, int]:
        """
        Ejecuta el procesamiento del examen y devuelve
        un resumen compatible con las estadísticas que espera process_batch().
        """
        # Ejecutar el thread de manera sincrónica
        exam_processor.run()          # ← bloqueante; si prefieres, usa .start()+.join()

        # El hilo llena exam_processor.stats (ver BatchProcessingThread.run)
        stats = exam_processor.stats

        return {
            'total':     stats.get('processed', 0),
            'successful': stats.get('successful', 0),
            'failed':    stats.get('failed', 0),
            'aligned':   stats.get('total_aligned', 0),
        }