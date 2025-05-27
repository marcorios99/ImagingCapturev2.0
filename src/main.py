#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path
import time
import os

from core.config import AppConfig 
from database.service import DatabaseService
from utils.logging_utils import setup_logging
from ui.cli_controller import CLIController

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Procesador de imágenes OMR/ICR en modo CLI')
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Número de workers para procesamiento paralelo'
    )
    parser.add_argument(
        '--filter', '-f',
        type=str,
        help='Archivo con lista de códigos de examen para filtrar'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['OMR', 'ICR'],
        help='Modo de procesamiento (OMR o ICR)'
    )
    parser.add_argument(
        '--loglevel',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Nivel de logging'
    )
    return parser.parse_args()

def print_header():
    """Imprime un header estético para la aplicación."""
    header = """
██╗███╗   ███╗ █████╗  ██████╗ ██╗███╗   ██╗ ██████╗     
██║████╗ ████║██╔══██╗██╔════╝ ██║████╗  ██║██╔════╝     
██║██╔████╔██║███████║██║  ███╗██║██╔██╗ ██║██║  ███╗    
██║██║╚██╔╝██║██╔══██║██║   ██║██║██║╚██╗██║██║   ██║    
██║██║ ╚═╝ ██║██║  ██║╚██████╔╝██║██║ ╚████║╚██████╔╝    
╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝     
                                                          
 ██████╗ █████╗ ██████╗ ████████╗██╗   ██╗██████╗ ███████╗
██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║   ██║██╔══██╗██╔════╝
██║     ███████║██████╔╝   ██║   ██║   ██║██████╔╝█████╗  
██║     ██╔══██║██╔═══╝    ██║   ██║   ██║██╔══██╗██╔══╝  
╚██████╗██║  ██║██║        ██║   ╚██████╔╝██║  ██║███████╗
 ╚═════╝╚═╝  ╚═╝╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
                                                           
v2.0.0 - Sistema de procesamiento de imágenes OMR/ICR
"""
    print(header)
    print("=" * 72)

def get_exam_codes_from_input():
    """Permite al usuario ingresar códigos de examen manualmente."""
    print("\nIngrese los códigos de examen (uno por línea).")
    print("Cuando termine, presione ENTER en una línea vacía:")
    print("-" * 40)
    
    codes = []
    while True:
        line = input().strip()
        if not line:
            break
        # Validar que sea numérico y agregar a la lista
        if line.isdigit():
            codes.append(line)
        else:
            print(f"Advertencia: '{line}' no es un código válido (debe ser numérico)")
    
    return codes

def save_codes_to_file(codes, filename="codigos_examen.txt"):
    """Guarda los códigos ingresados en un archivo de texto."""
    try:
        with open(filename, 'w') as f:
            for code in codes:
                f.write(f"{code}\n")
        print(f"Códigos guardados en: {filename}")
        return True
    except Exception as e:
        print(f"Error al guardar códigos en archivo: {e}")
        return False

class Application:
    def __init__(self):
        self.config = AppConfig()
        self.setup_logging()
        self.db_service = None
        self.cli_controller = None
        self.processing_mode = None

    def setup_logging(self):
        """Initialize logging configuration"""
        log_path = Path(self.config.LOGS_PATH) / "app.log" if hasattr(self.config, 'LOGS_PATH') else Path("logs/app.log")
        log_level = getattr(logging, args.loglevel)
        setup_logging(
            log_path=log_path,
            level=log_level,
            max_bytes=5_242_880,  # 5MB
            backup_count=5,
            json_format=False
        )
        self.logger = logging.getLogger(__name__)

    def initialize_database(self) -> bool:
        """Initialize database connection"""
        try:
            self.db_service = DatabaseService(self.config)
            self.db_service.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            print(f"Error: Failed to initialize database: {e}")
            return False

    def cleanup(self):
        """Perform cleanup operations"""
        try:
            if self.db_service:
                self.db_service.close()
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def select_processing_mode(self):
        """Permite al usuario seleccionar modo de procesamiento."""
        if args.mode:
            self.processing_mode = args.mode
            return True
            
        while True:
            print("\nSeleccione el modo de procesamiento:")
            print("1. OMR (Reconocimiento Óptico de Marcas)")
            print("2. ICR (Reconocimiento Inteligente de Caracteres)")
            print("0. Salir")
            
            option = input("\nOpción: ")
            
            if option == "1":
                self.processing_mode = "OMR"
                return True
            elif option == "2":
                self.processing_mode = "ICR"
                return True
            elif option == "0":
                return False
            else:
                print("Opción no válida. Intente nuevamente.")

    def get_exam_codes(self):
        """Obtiene los códigos de examen que se procesarán."""
        # Verificar si hay códigos desde un archivo de filtro
        if args.filter:
            try:
                with open(args.filter, 'r') as f:
                    exam_codes = [line.strip() for line in f if line.strip()]
                
                if exam_codes:
                    print(f"Cargados {len(exam_codes)} códigos de examen del archivo: {args.filter}")
                    return exam_codes
                else:
                    print(f"Advertencia: El archivo {args.filter} está vacío")
            except Exception as e:
                print(f"Error al leer archivo de códigos: {e}")
                self.logger.error(f"Error leyendo archivo {args.filter}: {e}")
        
        # Si no hay archivo o está vacío, mostrar el menú
        while True:
            print("\nSeleccione cómo ingresar los códigos de examen:")
            print("1. Ingresar códigos manualmente")
            print("2. Cargar códigos desde archivo .txt")
            print("0. Cancelar")
            
            option = input("\nOpción: ")
            
            if option == "1":
                codes = get_exam_codes_from_input()
                if codes:
                    # Ofrecer guardar los códigos en un archivo
                    if input("¿Desea guardar estos códigos en un archivo? (s/n): ").lower() == 's':
                        filename = input("Nombre del archivo (Enter para 'codigos_examen.txt'): ")
                        if not filename:
                            filename = "codigos_examen.txt"
                        save_codes_to_file(codes, filename)
                    return codes
                else:
                    print("No se ingresaron códigos válidos.")
            elif option == "2":
                filename = input("Ingrese la ruta al archivo .txt: ")
                try:
                    with open(filename, 'r') as f:
                        codes = [line.strip() for line in f if line.strip()]
                    if codes:
                        print(f"Cargados {len(codes)} códigos de examen")
                        return codes
                    else:
                        print("El archivo está vacío.")
                except Exception as e:
                    print(f"Error al leer el archivo: {e}")
            elif option == "0":
                return None
            else:
                print("Opción no válida. Intente nuevamente.")

    def run(self) -> int:
        """Main application execution"""
        try:
            print_header()
            print("Inicializando sistema...")
            
            if not self.initialize_database():
                print("Error: No se pudo conectar a la base de datos.")
                return 1
            
            print("Conexión a base de datos establecida correctamente.")
            
            # Seleccionar modo de procesamiento
            if not self.select_processing_mode():
                print("Operación cancelada por el usuario.")
                return 0
                
            print(f"\nModo de procesamiento seleccionado: {self.processing_mode}")
            
            # Obtener códigos de examen a procesar
            exam_codes = self.get_exam_codes()
            if not exam_codes:
                print("No se especificaron códigos de examen. Operación cancelada.")
                return 0
            
            print(f"\nSe procesarán {len(exam_codes)} códigos de examen.")
            
            # Inicializar el controlador CLI y filtrar por códigos
            self.cli_controller = CLIController(
                db_service=self.db_service,
                config=self.config
            )
            
            self.cli_controller.filtrar_por_codigos_examen(exam_codes)
            
            # Configurar modo de procesamiento
            # (Asumiendo que necesitas pasar este parámetro a alguna parte del sistema)
            self.config.processing_mode = self.processing_mode
            
            # Ejecutar procesamiento 
            print("\nIniciando procesamiento de imágenes...")
            start_time = time.time()
            self.cli_controller.run(workers=args.workers)
            
            # Mostrar resumen final
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print(f"\nProcesamiento completado en {minutes}m {seconds}s")
            
            self.cleanup()
            return 0
            
        except Exception as e:
            self.logger.critical(f"Fatal error: {e}", exc_info=True)
            print(f"Error fatal: {e}")
            return 1

def main():
    try:
        app = Application()
        return app.run()
    except KeyboardInterrupt:
        print("\nProcesamiento cancelado por el usuario")
        return 130  # 128 + SIGINT
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        print(f"Error fatal: {e}")
        return 1

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main())