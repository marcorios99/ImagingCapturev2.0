import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from core.config import AppConfig
from core.exceptions import (
    DatabaseError,
    ResourceNotFoundError,
    ValidationError
)
from database.connection import DatabaseConnection
from database.repository import Repository

class DatabaseService:
    """
    Servicio de acceso a datos con lógica de negocio completa para el procesamiento
    de fichas OMR/ICR.
    """
    
    def __init__(self, config: AppConfig):
        """
        Inicializa el servicio de base de datos.

        Args:
            config: Configuración de la aplicación
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection = DatabaseConnection(config)
        self.repository = Repository(self.connection)
        self._executor = None
        self._campos_cache = {}  # Cache simple para campos

        
    def initialize(self) -> None:
        """
        Inicializa las conexiones y recursos necesarios.
        
        Raises:
            DatabaseError: Si hay errores en la inicialización
        """
        try:
            # Inicializar conexión
            self.connection.initialize()
            
            # Configurar executor para operaciones asíncronas
            max_workers = self.config.max_threads
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="DBService"
            )
            
            # Verificar conexión
            conexion_info = self.repository.verificar_conexion()
            
            self.logger.info(
                f"Servicio de base de datos inicializado correctamente. "
                f"Base de datos: {conexion_info['base_datos']}, "
                f"Workers: {max_workers}"
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando servicio: {e}", exc_info=True)
            raise DatabaseError("Error al inicializar servicio de base de datos") from e
            
    def close(self) -> None:
        """
        Libera recursos y cierra conexiones.
        Esta función debe llamarse al terminar de usar el servicio.
        """
        try:
            self.clear_cache()
            
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
            
            self.connection.close()
            self.logger.info("Servicio de base de datos cerrado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error cerrando servicio: {e}", exc_info=True)
    
    def get_campos_plantilla(self, 
                        identificador: Union[int, str],
                        by_prefix: bool = False,
                        cache: bool = True) -> List[Dict[str, Any]]:
        """
        Obtiene los campos de una plantilla con lógica de negocio adicional.

        Args:
            identificador: ID de la plantilla o código de prefijo
            by_prefix: Si True, busca por prefijo. Si False, busca por ID
            cache: Si True, usa caché para resultados frecuentes

        Returns:
            List[Dict[str, Any]]: Lista de campos con su configuración

        Raises:
            ValidationError: Si los parámetros son inválidos
            DatabaseError: Si hay error en la consulta
            ResourceNotFoundError: Si no se encuentra la plantilla
        """
        try:
            # Generar clave de caché si está habilitado
            cache_key = f"campos_{'prefix' if by_prefix else 'id'}_{identificador}"
            
            # Verificar caché
            if cache and hasattr(self, '_campos_cache') and cache_key in self._campos_cache:
                self.logger.debug(f"Recuperado de caché: {cache_key}")
                return self._campos_cache[cache_key]

            # Obtener campos
            campos = self.repository.get_campos_plantilla(identificador, by_prefix)
            
            if not campos:
                criterio = "prefijo" if by_prefix else "ID"
                raise ResourceNotFoundError(
                    resource_type="Campos",
                    resource_id=str(identificador),
                    detail=f"No se encontraron campos para el {criterio}: {identificador}"
                )
                
            # Procesar y validar campos
            campos_procesados = []
            for campo in campos:
                # Validar coordenadas
                if not all(coord >= 0 for coord in [campo['x'], campo['y'], campo['ancho'], campo['alto']]):
                    self.logger.warning(
                        f"Campo {campo['campo_id']} tiene coordenadas inválidas: "
                        f"x={campo['x']}, y={campo['y']}, "
                        f"ancho={campo['ancho']}, alto={campo['alto']}"
                    )
                    continue
                
                # Agregar campo validado
                campos_procesados.append(campo)
                
            # Organizar campos por página
            campos_procesados.sort(key=lambda x: (x['pagina'], x['indice']))
            
            # Guardar en caché si está habilitado
            if cache:
                if not hasattr(self, '_campos_cache'):
                    self._campos_cache = {}
                self._campos_cache[cache_key] = campos_procesados
                self.logger.debug(f"Guardado en caché: {cache_key}")
                
            return campos_procesados
            
        except ResourceNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                f"Error recuperando campos para {identificador}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Error al obtener campos: {str(e)}") from e

    def clear_cache(self):
        """Limpia el caché de campos."""
        self._campos_cache.clear()
        self.logger.debug("Cache de campos limpiado")
            
            
    def get_plantilla_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """
        Obtiene una plantilla completa por su prefijo con validaciones adicionales.
        
        Args:
            prefix: Prefijo de tres dígitos
            
        Returns:
            Dict[str, Any]: Información completa de la plantilla
            
        Raises:
            ValidationError: Si el prefijo es inválido
            ResourceNotFoundError: Si no se encuentra la plantilla
            DatabaseError: Si hay error en la consulta
        """
        try:
            # Validar formato del prefijo
            if not prefix or not isinstance(prefix, str) or len(prefix) != 3 or not prefix.isdigit():
                raise ValidationError("El prefijo debe ser un string de 3 dígitos")
                
            # Obtener plantilla base
            plantilla = self.repository.get_plantilla_by_prefix(prefix)
            if not plantilla:
                raise ResourceNotFoundError(
                    resource_type="Plantilla",
                    resource_id=prefix,
                    detail=f"No se encontró plantilla para el prefijo: {prefix}"
                )
                
            return plantilla
            
        except ValidationError:
            raise
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error obteniendo plantilla para prefijo {prefix}: {e}")
            raise DatabaseError(f"Error al obtener plantilla: {str(e)}") from e
            
        
    def actualizar_estado_examen(self, examen_id: int, estado: str) -> None:
        """
        Actualiza el estado de un examen con validaciones adicionales.

        Args:
            examen_id: ID del examen a actualizar
            estado: Nuevo estado 
                   '0' - Pendiente
                   '1' - Procesado
                   '2' - Revisado

        Raises:
            ValidationError: Si el estado es inválido
            DatabaseError: Si hay error en la actualización
        """
        try:
            # Validar estado
            estados_validos = {'0', '1', '2'}
            if estado not in estados_validos:
                raise ValidationError(
                    f"Estado inválido. Debe ser uno de: {estados_validos}"
                )
                
            # Actualizar estado
            self.repository.actualizar_estado_examen(examen_id, estado)
            
            self.logger.info(
                f"Actualizado estado de examen {examen_id} a '{estado}'"
            )
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(
                f"Error actualizando estado de examen {examen_id}: {e}",
                exc_info=True
            )
            raise DatabaseError(
                f"Error al actualizar estado de examen: {str(e)}"
            ) from e

    def get_plantilla_by_id(self, plantilla_id: int) -> Dict[str, Any]:
        """
        Obtiene una plantilla específica con toda su información.
        
        Args:
            plantilla_id: ID de la plantilla
            
        Returns:
            Dict[str, Any]: Información completa de la plantilla
            
        Raises:
            ResourceNotFoundError: Si la plantilla no existe
            DatabaseError: Si hay errores al obtener los datos
        """
        try:
            plantillas = self.repository.get_plantillas()
            plantilla = next(
                (p for p in plantillas if p['plantilla_id'] == plantilla_id),
                None
            )
            
            if not plantilla:
                raise ResourceNotFoundError(
                    resource_type="Plantilla",
                    resource_id=str(plantilla_id)
                )

            plantilla['campos'] = self.repository.get_campos_plantilla(plantilla_id)

            return plantilla
            
        except ResourceNotFoundError:
            raise
        except Exception as e:
            self.logger.error(
                f"Error obteniendo plantilla {plantilla_id}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Error al obtener plantilla: {str(e)}") from e

    def crear_examen(self, codigo_barras: str, plantilla_id: int) -> int:
        """
        Crea un nuevo registro de examen.

        Args:
            codigo_barras: Código de barras único del examen
            plantilla_id: ID de la plantilla asociada

        Returns:
            int: ID del examen creado
        """
        try:
            # Validar que la plantilla exista
            plantilla = self.get_plantilla_by_id(plantilla_id)
            if not plantilla:
                raise ResourceNotFoundError(
                    resource_type="Plantilla",
                    resource_id=str(plantilla_id)
                )
            
            # Verificar si ya existe un examen con ese código
            examen_existente = self.buscar_examen_por_codigo(codigo_barras)
            if examen_existente:
                return examen_existente['examen_id']
            
            return self.repository.crear_examen(codigo_barras, plantilla_id)
            
        except Exception as e:
            self.logger.error(f"Error creando examen: {e}")
            raise DatabaseError(f"Error al crear examen: {str(e)}") from e

    def buscar_examen_por_codigo(self, codigo_barras: str) -> Optional[Dict[str, Any]]:
        """Busca un examen por su código de barras."""
        return self.repository.buscar_examen_por_codigo(codigo_barras)

    def registrar_imagen(self, exam_id: int, ruta: str, pagina: int, alineada: bool) -> int:
        """
        Registra una nueva imagen procesada.

        Args:
            exam_id: ID del examen asociado
            ruta: Ruta de la imagen
            pagina: Número de página
            alineada: Si la imagen está alineada

        Returns:
            int: ID de la imagen registrada
        """
        ruta_path = Path(ruta)
        if not ruta_path.exists():
            raise FileNotFoundError(f"No se encuentra el archivo: {ruta}")
        
        return self.repository.registrar_imagen(exam_id, str(ruta_path), pagina, alineada)

    def registrar_imagenes(self, imagenes: List[Dict[str, Any]]) -> List[int]:
        """
        Registra un lote de imágenes de forma optimizada.

        Args:
            imagenes: Lista de diccionarios con información de imágenes

        Returns:
            List[int]: Lista de IDs de las imágenes registradas
        """
        # Validar rutas de archivos
        for imagen in imagenes:
            if not Path(imagen['ruta']).exists():
                raise FileNotFoundError(f"No se encuentra el archivo: {imagen['ruta']}")
        
        return self.repository.registrar_imagenes(imagenes)

    def guardar_resultados(self, resultados: List[Dict[str, Any]]) -> None:
        """
        Guarda los resultados del procesamiento de imágenes.

        Args:
            resultados: Lista de resultados a guardar
        """
        # Validar datos mínimos requeridos
        for resultado in resultados:
            if not all(k in resultado for k in ['imagen_id', 'campo_id', 'certeza']):
                raise ValueError("Faltan campos requeridos en los resultados")
        
        self.repository.guardar_resultados(resultados)

    # Agregar los nuevos métodos aquí
    def guardar_resultado_icr(self, resultado: Dict[str, Any]) -> int:
        """
        Guarda un resultado ICR en la base de datos.
        
        Args:
            resultado: Diccionario con los datos del resultado
                - imagen_id: ID de la imagen
                - campo_id: ID del campo
                - valor: Valor detectado
                
        Returns:
            int: ID del resultado guardado
        """
        try:
            query = """
            INSERT INTO ResultadosIcr (ImagenID, CampoID, Valor, FechaProceso)
            OUTPUT INSERTED.ResultadoIcrID
            VALUES (?, ?, ?, GETDATE())
            """
            
            result = self.connection.fetch_one(
                query,
                (resultado['imagen_id'], resultado['campo_id'], resultado['valor'])
            )
            
            return int(result[0]) if result else 0
            
        except Exception as e:
            self.logger.error(f"Error guardando resultado ICR: {e}")
            raise DatabaseError(f"Error al guardar resultado ICR: {str(e)}") from e
            
    def guardar_resultados_procesamiento(self, resultados: List[Dict[str, Any]]) -> None:
        """
        Guarda resultados de procesamiento separando OMR e ICR.
        """
        try:
            resultados_omr = []
            resultados_icr = []
            
            for resultado in resultados:
                if resultado.get('tipo') in ['ICR', 'ICRN', 'ICRL']:
                    resultados_icr.append(resultado)
                else:
                    resultados_omr.append(resultado)
                    
            # Guardar resultados OMR
            if resultados_omr:
                self.guardar_resultados(resultados_omr)
                
            # Guardar resultados ICR
            for resultado in resultados_icr:
                self.guardar_resultado_icr({
                    'imagen_id': resultado['imagen_id'],
                    'campo_id': resultado['campo_id'],
                    'valor': resultado['valor']
                })
                
        except Exception as e:
            self.logger.error(f"Error guardando resultados de procesamiento: {e}")
            raise DatabaseError(str(e))
        
    def actualizar_estado_imagen(self, imagen_id: int, estado: str, alineado: bool = False) -> None:
        """
        Actualiza el estado de procesamiento de una imagen.

        Args:
            imagen_id: ID de la imagen
            estado: Nuevo estado ('0' - Pendiente, '1' - Procesado, '2' - Revisado, '3' - Finalizado)
            alineado: Indica si la imagen está alineada
        """
        estados_validos = {'0', '1', '2', '3'}
        if estado not in estados_validos:
            raise ValueError(f"Estado inválido. Debe ser uno de: {estados_validos}")
        
        self.repository.actualizar_estado_imagen(imagen_id, estado, alineado)

    def guardar_resultado_individual(
        self,
        resultado: Dict[str, Any],
        actualizar_estado: bool = True
    ) -> int:
        """
        Guarda un resultado individual de procesamiento y actualiza estados.
        
        Args:
            resultado: Información del resultado
            actualizar_estado: Si debe actualizar el estado de la imagen
            
        Returns:
            int: ID del resultado guardado
        """
        try:
            # Validar datos mínimos requeridos
            required_fields = ['imagen_id', 'campo_id', 'certeza']
            if not all(field in resultado for field in required_fields):
                raise ValueError("Faltan campos requeridos en el resultado")
            
            # Validar valores
            if not (0 <= resultado['certeza'] <= 100):
                raise ValueError("La certeza debe estar entre 0 y 100")
            
            # Guardar resultado y actualizar estado
            resultado_id = self.repository.guardar_resultado_procesamiento(resultado)
            
            if actualizar_estado:
                self.repository.actualizar_estado_imagen(
                    resultado['imagen_id'],
                    '1',  # Procesado
                    resultado.get('alineado', False)
                )
            
            return resultado_id
            
        except Exception as e:
            self.logger.error(
                f"Error guardando resultado individual: {e}",
                exc_info=True
            )
            raise DatabaseError(
                f"Error al guardar resultado: {str(e)}"
            ) from e
    
    
    def registrar_imagenes_examen(self, examen_id: int, imagenes: List[Dict[str, Any]]) -> List[int]:
        """
        Registra un conjunto de imágenes para un examen específico.
        
        Args:
            examen_id: ID del examen
            imagenes: Lista de diccionarios con información de imágenes
                Cada diccionario debe contener:
                - ruta: Path de la imagen
                - pagina: Número de página
                - alineada: (opcional) Si la imagen está alineada
                
        Returns:
            List[int]: Lista de IDs de las imágenes registradas (None para duplicados)
        """
        try:
            # Validar que todas las imágenes existan
            for imagen in imagenes:
                ruta_path = Path(imagen['ruta'])
                if not ruta_path.exists():
                    raise FileNotFoundError(f"No se encuentra el archivo: {ruta_path}")
                    
            # Registrar todas las imágenes en una sola transacción
            return self.repository.registrar_imagenes(examen_id, imagenes)
            
        except FileNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error en registro masivo: {e}")
            raise DatabaseError(f"Error al registrar imágenes: {str(e)}")

    def actualizar_resultado(self, campo_id: str, valor: Any) -> None:
        """Actualiza un resultado específico."""
        self.repository.actualizar_resultado(campo_id, valor)

    def eliminar_resultados_imagen(self, imagen_id: int) -> None:
        """Elimina resultados de una imagen y actualiza su estado."""
        self.repository.eliminar_resultados_imagen(imagen_id)

    def get_tipos_campo(self) -> List[Dict[str, Any]]:
        """Obtiene los tipos de campo disponibles."""
        return self.repository.get_tipos_campo()
    
    def examen_duplicado(self, codigo_barras: str) -> bool:
        """
        Verifica si un examen con el código de barras proporcionado
        ya existe en la base de datos.
        
        Returns:
            True si el examen existe, False en caso contrario.
        """
        examen_existente = self.buscar_examen_por_codigo(codigo_barras)
        return examen_existente is not None
    
    def get_imagenes_digitalizacion(self, exam_codes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Devuelve la lista de imágenes desde DIGITALIZACION.dbo.ImagenesProcesadas,
        filtrando opcionalmente por un listado de CodigoExamen.
        """
        return self.repository.get_imagenes_digitalizacion(exam_codes=exam_codes)

