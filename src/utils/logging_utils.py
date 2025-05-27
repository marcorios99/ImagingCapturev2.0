import logging, sys, json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console
import traceback


class CustomFormatter(logging.Formatter):
    """Custom formatter with different formats per log level."""
    def __init__(self):
        super().__init__()
        self.formatters = {
            logging.DEBUG: logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            logging.INFO: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'),
            logging.WARNING: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d'),
            logging.ERROR: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\nPath: %(pathname)s:%(lineno)d\nFunction: %(funcName)s'),
            logging.CRITICAL: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\nPath: %(pathname)s:%(lineno)d\nFunction: %(funcName)s\nProcess: %(process)d')
        }
        
    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.formatters[logging.DEBUG])
        if record.exc_info:
            record.exc_text = ''.join(traceback.format_exception(*record.exc_info))
        return formatter.format(record)
    
# utils/logging_utils.py
import logging, sys, json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console

# ────────────────────────────────────────────────────────────────
# Formatter “bonito” para consola con Rich no necesita nada más.
# Para archivo usamos tu CustomFormatter o JSON si lo pides.
# ────────────────────────────────────────────────────────────────

def setup_logging(
    log_path: Path,
    level: int = logging.INFO,
    max_bytes: int = 5_242_880,
    backup_count: int = 5,
    json_format: bool = False
) -> None:

    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # ─ Archivo rotativo ─
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(json.dumps({...})) if json_format else CustomFormatter()
    )
    root_logger.addHandler(file_handler)

    # ─ Consola Rich ─
    rich_handler = RichHandler(
        console=Console(),
        rich_tracebacks=True,
        markup=True,
        show_time=False
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(rich_handler)

    for lib in ["PIL", "opencv", "tensorflow", "matplotlib", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str, context: dict | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, context) if context else logger