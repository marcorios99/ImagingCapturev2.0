from typing import Optional, Any

class ApplicationError(Exception):
    """Excepción base para la aplicación."""
    
    def __init__(self, message: str, code: Optional[str] = None, 
                 details: Optional[Any] = None):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(self.message)

class DatabaseError(ApplicationError):
    """Error en operaciones de base de datos."""
    pass

class ProcessingError(ApplicationError):
    """Error durante el procesamiento de imágenes."""
    
    def __init__(self, message: str, image_path: Optional[str] = None, 
                 region: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.image_path = image_path
        self.region = region

class ValidationError(ApplicationError):
    """Error de validación de datos."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value

class ResourceNotFoundError(ApplicationError):
    """Error cuando no se encuentra un recurso."""
    
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        message = f"{resource_type} con id {resource_id} no encontrado"
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id

class ImageError(ApplicationError):
    """Error relacionado con imágenes."""
    
    def __init__(self, message: str, image_path: str, 
                 details: Optional[dict] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.image_path = image_path
        self.details = details or {}


def handle_exception(exc: Exception) -> ApplicationError:
    """
    Convierte excepciones genéricas en excepciones de la aplicación.
    Útil para manejar excepciones de bibliotecas externas.
    """
    if isinstance(exc, ApplicationError):
        return exc
        
    # Mapeo de excepciones comunes
    exception_mapping = {
        ValueError: ValidationError,
        FileNotFoundError: ResourceNotFoundError,
    }
    
    # Obtener la excepción correspondiente o usar ApplicationError por defecto
    exception_class = exception_mapping.get(type(exc), ApplicationError)
    
    return exception_class(
        message=str(exc),
        code='UNKNOWN_ERROR',
        details={
            'exception_type': type(exc).__name__,
            'original_message': str(exc)
        }
    )