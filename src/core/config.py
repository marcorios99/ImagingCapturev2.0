import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    """Configuración global de la aplicación."""
    APP_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    LOGS_PATH: Path = field(init=False)
    
    # Configuración de base de datos
    db_driver: str = "ODBC Driver 17 for SQL Server"
    db_server: str = "WINSERVER-04"
    db_name: str = "ENLA2024_CAPTURA"
    db_user: str = "sa"
    db_password: str = "safd.2024"
    db_trust_certificate: str = "yes"
    db_multiple_active_resultsets: str = "yes"
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_connection_timeout: int = 30
    db_query_timeout: int = 300
    db_pool_timeout: int = 30
    db_connection_lifetime = 0
    db_connection_reset = "Yes"
    
    # Configuración de procesamiento
    batch_size: int = 10
    max_threads: int = os.cpu_count() or 4
    dpi_min: int = 300
    omr_threshold: float = 230.0
    icr_threshold: float = 0
    min_area: int = 900000
    timeout_seconds: int = 30
    max_retries: int = 3
    
    # Configuración de logging
    log_level: int = logging.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    
    def __post_init__(self):
        """Inicializa rutas y carga configuración."""
        self.LOGS_PATH = self.APP_ROOT / "logs"

    