import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging
import json
import platform
import pkg_resources
from logging.handlers import RotatingFileHandler

def setup_logging(log_path, max_bytes=5*1024*1024, backup_count=3):
    """Configura el sistema de logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_build_logging():
    """Configura logging para el proceso de build."""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return setup_logging(
            log_path=log_dir / "build.log",
            max_bytes=5*1024*1024,
            backup_count=3
        )
    except Exception as e:
        print(f"Error configurando logging de build: {e}")
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

def get_package_version(package_name):
    """Obtiene la versión instalada de un paquete."""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def check_dependencies():
    """Verifica las dependencias requeridas"""
    print("Verificando dependencias...")
    
    required_packages = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'pyodbc': 'pyodbc',
        'PySide6': 'PySide6',
        'scipy': 'scipy',
        'zxing-cpp': 'zxingcpp',
        'XlsxWriter': 'xlsxwriter',
        'tensorflow': 'tensorflow',
        'h5py': 'h5py',
        'pandas': 'pandas',
        'Pillow': 'PIL',
        'packaging': 'packaging'
    }
    
    missing_packages = []
    installed_versions = {}
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            version = get_package_version(package)
            installed_versions[package] = version
            print(f"✓ {package} ({version})")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print("\nFaltan las siguientes dependencias:")
        print("\n".join(missing_packages))
        print("\nInstale las dependencias con:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_system_requirements():
    """Verifica los requisitos del sistema"""
    print("\nVerificando requisitos del sistema...")
    
    requirements_met = True
    
    # Verificar Sistema Operativo
    os_info = platform.system(), platform.release(), platform.architecture()
    print(f"Sistema Operativo: {os_info[0]} {os_info[1]} {os_info[2][0]}")
    if os_info[0] != "Windows":
        print("✗ Se requiere Windows")
        requirements_met = False
    
    # Verificar Python
    python_version = sys.version_info
    if python_version < (3, 8):
        print("✗ Se requiere Python 3.8 o superior")
        requirements_met = False
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Verificar arquitectura
    if platform.architecture()[0] != '64bit':
        print("✗ Se requiere sistema operativo de 64 bits")
        requirements_met = False
    else:
        print("✓ Sistema 64 bits")
    
    # Verificar ODBC
    try:
        import pyodbc
        drivers = pyodbc.drivers()
        if 'ODBC Driver 17 for SQL Server' not in drivers:
            print("✗ ODBC Driver 17 no encontrado")
            requirements_met = False
        else:
            print("✓ ODBC Driver 17")
        print("Drivers ODBC disponibles:")
        for driver in drivers:
            print(f"  - {driver}")
    except Exception as e:
        print(f"✗ Error verificando ODBC: {e}")
        requirements_met = False
    
    return requirements_met

def clean_build_directory():
    """Limpia los directorios de build"""
    print("Limpiando directorios de build...")
    directories = ['build', 'dist']
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Eliminado directorio: {dir_name}")
        dir_path.mkdir(parents=True)
        print(f"Creado directorio: {dir_name}")

def find_dll_path(module_name):
    """Encuentra la ruta de las DLLs de un módulo."""
    try:
        module = __import__(module_name)
        return os.path.dirname(module.__file__)
    except ImportError:
        return None

def copy_dlls():
    """Copia todas las DLLs necesarias"""
    print("\nCopiando DLLs necesarias...")
    dist_dir = Path("dist")
    
    # Mapeo de módulos y sus DLLs
    dll_modules = {
        'cv2': 'OpenCV',
        'zxingcpp': 'ZXing',
        'tensorflow': 'TensorFlow'
    }
    
    for module_name, description in dll_modules.items():
        try:
            module_path = find_dll_path(module_name)
            if module_path:
                dlls = [f for f in os.listdir(module_path) if f.endswith('.dll')]
                for dll in dlls:
                    src = os.path.join(module_path, dll)
                    dst = dist_dir / dll
                    shutil.copy2(src, dst)
                    print(f"Copiada DLL de {description}: {dll}")
        except Exception as e:
            print(f"Error copiando DLLs de {description}: {e}")

def copy_additional_files():
    """Copia archivos adicionales necesarios"""
    print("\nCopiando archivos adicionales...")
    dist_dir = Path("dist")
    
    # Estructura de directorios
    directories = [
        'installers',
        'resources',
        'database/scripts',
        'logs',
        'temp',
        'data',
        'config'
    ]
    
    for directory in directories:
        dir_path = dist_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Creado directorio: {directory}")
    
    # Archivos a copiar
    files_to_copy = [
        ("installers/msodbcsql.msi", "installers/msodbcsql.msi"),
        ("installers/VC_redist.x64.exe", "installers/VC_redist.x64.exe"),
        ("LICENSE", "LICENSE"),
        ("README.md", "README.md"),
        ("version_info.txt", "version_info.txt")
    ]
    
    for src, dst in files_to_copy:
        try:
            src_path = Path(src)
            if src_path.exists():
                dst_path = dist_dir / dst
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"Copiado: {src}")
            else:
                print(f"Advertencia: No se encuentra el archivo {src}")
        except Exception as e:
            print(f"Error copiando {src}: {e}")

def create_version_info():
    """Crea el archivo de información de versión"""
    print("\nCreando archivo de información de versión...")
    version_info = """
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'SAFD'),
         StringStruct(u'FileDescription', u'IMAGING-CAPTURE'),
         StringStruct(u'FileVersion', u'1.0.0'),
         StringStruct(u'InternalName', u'IMAGING-CAPTURE'),
         StringStruct(u'LegalCopyright', u'Copyright (c) 2024 SAFD'),
         StringStruct(u'OriginalFilename', u'IMAGING-CAPTURE.exe'),
         StringStruct(u'ProductName', u'IMAGING-CAPTURE'),
         StringStruct(u'ProductVersion', u'1.0.0')])
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    try:
        with open('version_info.txt', 'w', encoding='utf-8') as f:
            f.write(version_info.strip())
        print("✓ Archivo de versión creado exitosamente")
    except Exception as e:
        print(f"✗ Error creando archivo de versión: {e}")
        raise

def verify_deployment_config():
    """Verifica que el archivo deployment.json existe y es válido"""
    print("\nVerificando configuración de deployment...")
    try:
        if not Path("deployment.json").exists():
            raise FileNotFoundError("No se encuentra el archivo deployment.json")
        
        with open("deployment.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        required_fields = [
            "app_name", "version", "identifier", "script", 
            "hidden_imports", "include_files"
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Campo requerido faltante en deployment.json: {field}")
        
        print("✓ Configuración de deployment válida")
        return True
    except Exception as e:
        print(f"✗ Error en configuración de deployment: {e}")
        return False

def run_deployment():
    """Ejecuta el proceso de deployment con PySide6"""
    print("\nEjecutando proceso de deployment...")
    try:
        deploy_args = [
            'pyside6-deploy',
            'main.py',
            '--config', 'deployment.json',
            '--verbose',
            '--force'
        ]
        
        result = subprocess.run(
            deploy_args,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("Advertencias durante el deployment:")
            print(result.stderr)
        
        print("✓ Deployment completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error en el proceso de deployment: {e}")
        if e.stdout:
            print("Salida del proceso:")
            print(e.stdout)
        if e.stderr:
            print("Errores del proceso:")
            print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ Error inesperado durante el deployment: {e}")
        return False


def build():
    """Proceso principal de build"""
    logger = setup_build_logging()
    logger.info("Iniciando proceso de build...")
    
    try:
        # Verificar requisitos previos
        if not check_dependencies():
            logger.error("Faltan dependencias requeridas")
            return False
        
        if not check_system_requirements():
            logger.error("No se cumplen los requisitos del sistema")
            return False
        
        if not verify_deployment_config():
            logger.error("Configuración de deployment inválida")
            return False
        
        # Limpiar y preparar directorios
        clean_build_directory()
        
        # Crear archivo de versión
        create_version_info()
        
        # Ejecutar deployment
        if not run_deployment():
            logger.error("Falló el proceso de deployment")
            return False
        
        # Copiar archivos adicionales
        copy_dlls()
        copy_additional_files()
        
        logger.info("Build completado exitosamente!")
        print("\n✓ Build completado exitosamente!")
        print("Los archivos se encuentran en el directorio 'dist'")
        return True
        
    except Exception as e:
        logger.error(f"Error en el proceso de build: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return False
    finally:
        logging.shutdown()

if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)