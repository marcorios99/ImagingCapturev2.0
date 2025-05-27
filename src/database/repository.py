import logging
from typing import List, Dict, Optional, Any, Union
from core.exceptions import DatabaseError, ValidationError
from database.connection import DatabaseConnection
from typing import List, Dict, Any, Optional, Set, Union

class Repository:
    """Repositorio optimizado para procesamiento de fichas OMR/ICR."""
    
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
        self.logger = logging.getLogger(__name__)

    def get_plantillas(self) -> List[Dict[str, Any]]:
        """
        Obtiene todas las plantillas disponibles con información básica.

        Returns:
            List[Dict[str, Any]]: Lista de plantillas con sus atributos básicos
        """
        try:
            query = """
            SELECT 
                p.PlantillaID,
                p.Nombre,
                p.RutaImagen,
                p.NroPaginas,
                p.FechaCreacion,
                p.FechaModificacion,
                COUNT(c.CampoID) as TotalCampos
            FROM Plantilla p
            LEFT JOIN Campo c ON p.PlantillaID = c.PlantillaID
            GROUP BY 
                p.PlantillaID, p.Nombre, p.RutaImagen,
                p.NroPaginas, p.FechaCreacion, p.FechaModificacion
            ORDER BY p.Nombre
            """
            
            results = self.connection.fetch_all(query)
            return [{
                'plantilla_id': row[0],
                'nombre': row[1],
                'ruta_imagen': row[2],
                'nro_paginas': row[3],
                'fecha_creacion': row[4],
                'fecha_modificacion': row[5],
                'total_campos': row[6]
            } for row in results]
            
        except Exception as e:
            self.logger.error(f"Error obteniendo plantillas: {e}")
            raise DatabaseError(f"Error al obtener plantillas: {str(e)}") from e

    def get_plantilla_by_prefix(self, prefix: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una plantilla por su prefijo asociado.

        Args:
            prefix: Prefijo a buscar (3 dígitos)

        Returns:
            Optional[Dict[str, Any]]: Información de la plantilla o None si no existe

        Raises:
            DatabaseError: Si hay error en la consulta
        """
        try:
            query = """
            SELECT
                p.PlantillaID,
                p.Nombre,
                p.Prefijo,
                p.RutaImagen,
                p.NroPaginas,
                p.PaginasVariables,
                p.FechaCreacion,
                p.FechaModificacion
            FROM Plantilla p
            WHERE p.Prefijo = ?
            """
                        
            result = self.connection.fetch_one(query, (prefix,))
            
            if result:
                return {
                    'plantilla_id': result[0],
                    'nombre': result[1],
                    'prefijo': result[2],
                    'ruta_imagen': result[3],
                    'nro_paginas': result[4],
                    'paginas_variables': bool(result[5]),
                    'fecha_creacion': result[6],
                    'fecha_modificacion': result[7]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error obteniendo plantilla por prefijo {prefix}: {e}")
            raise DatabaseError(f"Error al obtener plantilla: {str(e)}") from e


    def get_campos_plantilla(self, 
                       identificador: Union[int, str],
                       by_prefix: bool = False) -> List[Dict[str, Any]]:
        """
        Obtiene todos los campos definidos para una plantilla específica.

        Args:
            identificador: ID de la plantilla (int) o prefijo (str)
            by_prefix: Si True, busca por prefijo. Si False, busca por ID.

        Returns:
            List[Dict[str, Any]]: Lista de campos con sus configuraciones
            
        Raises:
            DatabaseError: Si hay error en la consulta
            ValidationError: Si los parámetros son inválidos
        """
        try:
            query = """
            SELECT 
                c.CampoID,
                c.Pregunta,
                tc.Nombre as TipoCampo,
                c.PaginaNumero,
                c.X,
                c.Y,
                c.Ancho,
                c.Alto,
                c.UmbralLocal,
                c.Validaciones,
                c.Indice,
                c.Longitud
            FROM Campo c
            INNER JOIN TipoCampo tc ON c.TipoCampoID = tc.TipoCampoID
            INNER JOIN Plantilla p ON c.PlantillaID = p.PlantillaID
            WHERE {where_clause}
            ORDER BY c.PaginaNumero, c.Indice
            """
            
            if by_prefix:
                if not isinstance(identificador, str):
                    raise ValidationError("El prefijo debe ser un string")
                query = query.format(where_clause="p.Prefijo = ?")
            else:
                if not isinstance(identificador, (int, str)) or (isinstance(identificador, str) and not identificador.isdigit()):
                    raise ValidationError("El ID debe ser un número")
                query = query.format(where_clause="c.PlantillaID = ?")
                identificador = int(identificador)
            
            results = self.connection.fetch_all(query, (identificador,))
            
            if not results:
                criterio = "prefijo" if by_prefix else "ID"
                raise ValidationError(f"No se encontraron campos para el {criterio}: {identificador}")
            
            return [{
                'campo_id': row[0],
                'pregunta': row[1],
                'tipo_campo': row[2],
                'pagina': row[3],
                'x': row[4],
                'y': row[5],
                'ancho': row[6],
                'alto': row[7],
                'umbral_local': row[8],
                'validaciones': row[9],
                'indice': row[10],
                'longitud': row[11]
            } for row in results]
            
        except ValidationError:
            raise
        except Exception as e:
            criterio = "prefijo" if by_prefix else "ID"
            self.logger.error(f"Error obteniendo campos por {criterio}: {identificador}", exc_info=True)
            raise DatabaseError(f"Error al obtener campos: {str(e)}") from e

    def get_tipos_campo(self) -> List[Dict[str, Any]]:
        """
        Obtiene todos los tipos de campo disponibles.

        Returns:
            List[Dict[str, Any]]: Lista de tipos de campo
        """
        try:
            query = """
            SELECT 
                TipoCampoID,
                Nombre,
                Descripcion
            FROM TipoCampo
            ORDER BY Nombre
            """
            
            results = self.connection.fetch_all(query)
            return [{
                'tipo_campo_id': row[0],
                'nombre': row[1],
                'descripcion': row[2]
            } for row in results]
            
        except Exception as e:
            self.logger.error(f"Error obteniendo tipos de campo: {e}")
            raise DatabaseError(f"Error al obtener tipos de campo: {str(e)}") from e

    def get_all_barcodes(self, batch_size: int = 1000) -> Set[str]:
        """
        Obtiene todos los códigos de barra existentes en la tabla Examen.
        Implementa lectura por lotes para optimizar memoria.

        Args:
            batch_size: Tamaño de lote para cada consulta

        Returns:
            Set[str]: Conjunto de códigos de barra únicos
        """
        try:
            # Obtener número total de códigos para planificar lotes
            count_query = "SELECT COUNT(CodigoBarras) FROM Examen"
            result = self.connection.fetch_one(count_query)
            total_count = result[0] if result else 0
            
            if total_count == 0:
                return set()
                
            # Inicializar conjunto para códigos únicos
            all_barcodes = set()
            
            # Usar paginación para cargar por lotes
            for offset in range(0, total_count, batch_size):
                query = """
                    SELECT CodigoBarras
                    FROM Examen
                    ORDER BY ExamenID
                    OFFSET ? ROWS
                    FETCH NEXT ? ROWS ONLY
                """
                results = self.connection.fetch_all(query, (offset, batch_size))

                all_barcodes.update(row[0] for row in results if row[0])

                loaded = min(offset + batch_size, total_count)
                self.logger.debug(f"Cargados {len(all_barcodes)} códigos de barra ({loaded}/{total_count})")
                
                # (Opcional) liberar memoria después de registrar el log
                del results
                
            return all_barcodes
                
        except Exception as e:
            self.logger.error(f"Error obteniendo códigos de barra: {e}")
            raise DatabaseError(f"Error al obtener códigos de barra: {str(e)}") from e


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
            query = """
            INSERT INTO Examen (CodigoBarras, PlantillaID, Estado)
            OUTPUT INSERTED.ExamenID
            VALUES (?, ?, '0')
            """
            
            result = self.connection.fetch_one(query, (codigo_barras, plantilla_id))
            return int(result[0]) if result else 0
            
        except Exception as e:
            self.logger.error(f"Error creando examen: {e}")
            raise DatabaseError(f"Error al crear examen: {str(e)}") from e

    def buscar_examen_por_codigo(self, codigo_barras: str) -> Optional[Dict[str, Any]]:
        """
        Busca un examen por su código de barras.
        
        Args:
            codigo_barras: Código de barras a buscar

        Returns:
            Optional[Dict[str, Any]]: Información del examen si existe
        """
        try:
            query = """
            SELECT ExamenID, PlantillaID, Estado
            FROM Examen
            WHERE CodigoBarras = ?
            """
            
            result = self.connection.fetch_one(query, (codigo_barras,))
            
            if result:
                return {
                    'examen_id': result[0],
                    'plantilla_id': result[1],
                    'estado': result[2]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error buscando examen: {e}")
            raise DatabaseError(f"Error al buscar examen: {str(e)}")

    def registrar_imagen(self, examen_id: int, ruta: str, pagina: int, alineada: bool) -> int:
        try:
            query = """
            INSERT INTO Imagen (
                ExamenID, RutaImagen, NumeroPagina, 
                Alineado, Estado, FechaCreacion
            ) 
            OUTPUT INSERTED.ImagenID
            VALUES (?, ?, ?, ?, '1', GETDATE())
            """
            
            result = self.connection.fetch_one(
                query, 
                (examen_id, ruta, pagina, alineada)
            )

            return int(result[0]) if result else 0
            
        except Exception as e:
            self.logger.error(f"Error registrando imagen: {e}")
            raise DatabaseError(f"Error al registrar imagen: {str(e)}")

    def guardar_resultados(self, resultados: List[Dict[str, Any]]) -> None:
        """
        Guarda los resultados del procesamiento de imágenes.

        Args:
            resultados: Lista de resultados a guardar
        """
        try:
            query = """
            INSERT INTO Resultados 
                (ImagenID, CampoID, Certeza, FechaProceso)
            VALUES (?, ?, ?, GETDATE())
            """
            
            params = [(
                r['imagen_id'],
                r['campo_id'],
                r['certeza']
            ) for r in resultados]
            
            self.connection.execute_many(query, params)
            
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {e}")
            raise DatabaseError(f"Error al guardar resultados: {str(e)}") from e

    def actualizar_estado_imagen(self, imagen_id: int, estado: str, alineado: bool = False) -> None:
        """
        Actualiza el estado de procesamiento de una imagen.

        Args:
            imagen_id: ID de la imagen
            estado: Nuevo estado ('0' - Pendiente, '1' - Procesado, '2' - Revisado, '3' - Finalizado)
            alineado: Indica si la imagen está alineada
        """
        try:
            query = """
            UPDATE Imagen 
            SET Estado = ?, Alineado = ?
            WHERE ImagenID = ?
            """
            
            self.connection.execute_query(query, (estado, alineado, imagen_id))
            
        except Exception as e:
            self.logger.error(f"Error actualizando estado de imagen: {e}")
            raise DatabaseError(f"Error al actualizar estado: {str(e)}") from e


    def eliminar_resultados_imagen(self, imagen_id: int) -> None:
        """
        Elimina todos los resultados asociados a una imagen.

        Args:
            imagen_id: ID de la imagen
        """
        try:
            query = """
            DELETE FROM Resultados
            WHERE ImagenID = ?
            """
            
            self.connection.execute_query(query, (imagen_id,))
            
            # Actualizar estado de la imagen
            self.actualizar_estado_imagen(imagen_id, '0', False)
            
        except Exception as e:
            self.logger.error(f"Error eliminando resultados: {e}")
            raise DatabaseError(f"Error al eliminar resultados: {str(e)}") from e

   
    def guardar_resultado_procesamiento(self, resultado: Dict[str, Any]) -> int:
        try:
            with self.connection.transaction() as cursor:
                # Insertar resultado principal
                query = """
                INSERT INTO Resultados 
                    (ImagenID, CampoID, Certeza, FechaProceso)
                OUTPUT INSERTED.ResultadoID
                VALUES (?, ?, ?, GETDATE())
                """
                
                cursor.execute(query, (
                    resultado['imagen_id'],
                    resultado['campo_id'],
                    resultado['certeza']
                ))
                
                result = cursor.fetchone()
                resultado_id = int(result[0]) if result else 0
                
                # Actualizar estado de la imagen si es necesario
                if resultado.get('actualizar_estado', True):
                    self.actualizar_estado_imagen(
                        resultado['imagen_id'], 
                        '1',  # Procesado
                        resultado.get('alineado', False)
                    )
                
                return resultado_id
                
        except Exception as e:
            self.logger.error(f"Error guardando resultado: {e}")
            raise DatabaseError(f"Error al guardar resultado: {str(e)}") from e
        
        
    def actualizar_estado_examen(self, examen_id: int, estado: str) -> None:
        """
        Actualiza el estado de un examen.

        Args:
            examen_id: ID del examen
            estado: Nuevo estado ('0' - Pendiente, '1' - Procesado, '2' - Revisado)

        Raises:
            DatabaseError: Si hay error en la actualización
        """
        try:
            query = """
            UPDATE Examen 
            SET 
                Estado = ?,
                FechaModificacion = GETDATE()
            WHERE ExamenID = ?
            """
            
            rows_affected = self.connection.execute_query(query, (estado, examen_id))
            
            if rows_affected == 0:
                self.logger.warning(f"No se encontró examen con ID: {examen_id}")
                raise ValidationError(f"No existe examen con ID: {examen_id}")
                
        except Exception as e:
            self.logger.error(f"Error actualizando estado de examen {examen_id}: {e}")
            raise DatabaseError(f"Error al actualizar estado de examen: {str(e)}") from e

    def actualizar_resultado(self, campo_id: str, valor: Any) -> None:
        """
        Actualiza un resultado específico en la base de datos.

        Args:
            campo_id: ID del campo a actualizar
            valor: Nuevo valor del resultado
        """
        try:
            query = """
            UPDATE Resultados
            SET Certeza = ?
            WHERE CampoID = ?
            """
            self.connection.execute_query(query, (valor, campo_id))
        except Exception as e:
            self.logger.error(f"Error actualizando resultado: {e}")
            raise DatabaseError(f"Error al actualizar resultado: {str(e)}")

    def verificar_conexion(self) -> Dict[str, Any]:
        """
        Verifica el estado de la conexión a la base de datos.

        Returns:
            Dict[str, Any]: Información de la conexión
                - estado: True si la conexión está activa
                - base_datos: Nombre de la base de datos
                - hora_servidor: Hora actual del servidor
                - version: Versión del servidor

        Raises:
            DatabaseError: Si hay problemas con la conexión
        """
        try:
            query = """
            SELECT 1 AS connection_test,
                   DB_NAME() AS current_database,
                   GETDATE() AS server_time,
                   @@VERSION AS server_version
            """
            
            result = self.connection.fetch_one(query)
            
            if not result or result[0] != 1:
                raise DatabaseError("Prueba de conexión fallida")
                
            return {
                'estado': True,
                'base_datos': result[1],
                'hora_servidor': result[2],
                'version': result[3]
            }
            
        except Exception as e:
            self.logger.error(f"Error verificando conexión: {e}")
            raise DatabaseError(f"Error al verificar conexión: {str(e)}") from e
        

    def get_existing_images(self, examen_id: int) -> Dict[str, Any]:
        """
        Obtiene todas las imágenes existentes para un examen.
        
        Args:
            examen_id: ID del examen
            
        Returns:
            Dict con información de imágenes existentes:
            - 'rutas': Set de rutas de imágenes
            - 'paginas': Dict de número de página por ruta
        """
        try:
            query = """
            SELECT RutaImagen, NumeroPagina
            FROM Imagen 
            WHERE ExamenID = ?
            """
            
            results = self.connection.fetch_all(query, (examen_id,))
            
            existing_paths = set()
            page_numbers = {}
            
            for ruta, pagina in results:
                # Normalizar la ruta para comparación
                ruta_normalizada = str(ruta).lower().replace('\\', '/')
                existing_paths.add(ruta_normalizada)
                page_numbers[ruta_normalizada] = pagina
                
            return {
                'rutas': existing_paths,
                'paginas': page_numbers
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo imágenes existentes: {e}")
            raise DatabaseError(f"Error al obtener imágenes existentes: {str(e)}")

    def registrar_imagenes(self, examen_id: int, imagenes: List[Dict[str, Any]]) -> List[int]:
        """
        Registra un lote de imágenes evitando duplicados de manera eficiente.
        """
        if not imagenes:
            return []

        try:
            # Obtener imágenes existentes una sola vez
            existing_data = self.get_existing_images(examen_id)
            
            with self.connection.transaction() as cursor:
                image_ids = []
                
                # Preparar la consulta de inserción
                insert_query = """
                INSERT INTO Imagen 
                    (ExamenID, RutaImagen, NumeroPagina, Alineado, Estado, FechaCreacion)
                OUTPUT INSERTED.ImagenID
                VALUES (?, ?, ?, ?, '1', GETDATE())
                """
                
                for imagen in imagenes:
                    # Normalizar la ruta para comparación
                    ruta_original = str(imagen['ruta'])
                    ruta_normalizada = ruta_original.lower().replace('\\', '/')
                    pagina = imagen['pagina']
                    
                    # Verificar si la imagen ya existe para este examen
                    if ruta_normalizada in existing_data['rutas']:
                        self.logger.warning(
                            f"Imagen duplicada omitida - "
                            f"Examen: {examen_id}, "
                            f"Ruta: {ruta_original}, "
                            f"Página existente: {existing_data['paginas'].get(ruta_normalizada)}"
                        )
                        image_ids.append(None)
                        continue
                    
                    # Si no es duplicado, insertar y actualizar sets
                    try:
                        cursor.execute(insert_query, (
                            examen_id,
                            ruta_original,
                            pagina,
                            imagen.get('alineada', False)
                        ))
                        result = cursor.fetchone()
                        image_id = int(result[0]) if result else None
                        
                        if image_id:
                            existing_data['rutas'].add(ruta_normalizada)
                            existing_data['paginas'][ruta_normalizada] = pagina
                            
                        image_ids.append(image_id)
                        
                    except Exception as ex:
                        self.logger.error(f"Error insertando imagen {ruta_original}: {ex}")
                        image_ids.append(None)
                
                return image_ids
                
        except Exception as e:
            self.logger.error(f"Error en registro masivo de imágenes: {e}")
            raise DatabaseError(f"Error en registro masivo: {str(e)}")
        

    def get_imagenes_digitalizacion(
        self,
        exam_codes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene las imágenes de la tabla DIGITALIZACION.dbo.ImagenesProcesadas.
        Si 'exam_codes' no es None (ni vacío), filtra únicamente por esos códigos.
        """
        try:
            base_query = """
                SELECT 
                    BarcodeC39,
                    Ruta,
                    CodigoExamen,
                    Caja,
                    Pallet
                FROM DIGITALIZACION.dbo.ImagenesProcesadas
            """
            
            # Si no hay exam_codes, usamos la consulta "completa"
            if not exam_codes:
                query = base_query
                params = ()
            else:
                # Construir placeholders para la cláusula IN
                # e.g. si exam_codes = ['ABC','DEF'], => placeholders = (?,?)
                placeholders = ','.join(['?' for _ in exam_codes])
                query = base_query + f" WHERE CodigoExamen IN ({placeholders})"
                params = tuple(exam_codes)
            
            results = self.connection.fetch_all(query, params)
            
            imagenes = []
            for row in results:
                # row = (BarcodeC39, Ruta, CodigoExamen, Caja, Pallet)
                imagenes.append({
                    'barcode_c39': row[0],
                    'ruta': row[1],
                    'codigo_examen': row[2],
                    'caja': row[3],
                    'pallet': row[4]
                })
            return imagenes
        
        except Exception as e:
            self.logger.error(f"Error obteniendo imágenes de digitalización: {e}")
            raise DatabaseError(f"Error al obtener imágenes de digitalización: {str(e)}") from e
