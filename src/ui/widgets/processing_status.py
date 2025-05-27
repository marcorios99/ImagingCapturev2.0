import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QTableWidget, QTableWidgetItem,
    QFrame, QHeaderView
)
from PySide6.QtCore import Qt, QDateTime, Signal
from PySide6.QtGui import QColor, QBrush

@dataclass
class ProcessingState:
    """Estado del procesamiento."""
    total_images: int = 0
    processed: int = 0
    success: int = 0
    errors: int = 0
    start_time: Optional[QDateTime] = None

class ProcessingStatusWidget(QWidget):
    """Widget simplificado para mostrar el estado del procesamiento en tiempo real."""
    
    # Señales
    status_updated = Signal(dict)
    processing_stopped = Signal()
    
    # Estados de resultados
    RESULT_COLORS = {
        'success': QColor(200, 255, 200),  # Verde claro
        'error': QColor(255, 200, 200),    # Rojo claro
        'warning': QColor(255, 255, 200)   # Amarillo claro
    }
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.state = ProcessingState()
        self._setup_ui()
        
    def _setup_ui(self):
        """Configura la interfaz del widget."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Panel superior con progreso
        progress_panel = self._create_progress_panel()
        layout.addWidget(progress_panel)
        
        # Tabla de resultados
        self.results_table = self._create_results_table()
        layout.addWidget(self.results_table, stretch=1)
        
    def _create_progress_panel(self) -> QFrame:
        """Crea el panel de progreso."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(panel)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v/%m)")
        
        # Etiqueta de estado
        self.status_label = QLabel("Preparado")
        self.status_label.setStyleSheet("font-weight: bold;")
        
        layout.addWidget(QLabel("Progreso:"))
        layout.addWidget(self.progress_bar, stretch=1)
        layout.addWidget(self.status_label)
        
        return panel
    
    def _create_results_table(self) -> QTableWidget:
        """Crea la tabla de resultados."""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels([
            "Archivo", "Estado", "Tiempo", "Detalles"
        ])
        
        # Optimizar rendimiento
        table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        
        # Configurar columnas
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Archivo
        header.setSectionResizeMode(1, QHeaderView.Fixed)    # Estado
        header.setSectionResizeMode(2, QHeaderView.Fixed)    # Tiempo
        header.setSectionResizeMode(3, QHeaderView.Stretch)  # Detalles
        
        table.setColumnWidth(1, 100)
        table.setColumnWidth(2, 80)
        
        # Optimizaciones
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        
        return table
    
    def reset(self):
        """Resetea el widget a su estado inicial."""
        try:
            # Resetear estado
            self.state = ProcessingState()
            
            # Resetear UI
            self.progress_bar.setValue(0)
            self.status_label.setText("Preparado")
            self.results_table.setRowCount(0)
            
        except Exception as e:
            self.logger.error(f"Error reseteando widget: {e}")
    
    def start_processing(self, total_images: int):
        """Inicia el monitoreo de procesamiento."""
        if total_images <= 0:
            raise ValueError("El número total de imágenes debe ser mayor a 0")
            
        # Resetear estado
        self.state = ProcessingState(
            total_images=total_images,
            start_time=QDateTime.currentDateTime()
        )
        
        # Actualizar UI
        self.progress_bar.setMaximum(total_images)
        self.progress_bar.setValue(0)
        self.status_label.setText("Procesando...")
        self.results_table.setRowCount(0)
        
    def update_progress(self, result: Dict[str, Any]):
        """Actualiza el progreso con un nuevo resultado."""
        try:
            # Actualizar estado
            self.state.processed += 1
            if result.get('status') == 'success':
                self.state.success += 1
            elif result.get('status') == 'error':
                self.state.errors += 1
                
            # Actualizar barra de progreso
            self.progress_bar.setValue(self.state.processed)
            
            # Agregar resultado a la tabla
            self._add_result_to_table(result)
            
            # Notificar actualización
            self.status_updated.emit({
                'processed': self.state.processed,
                'success': self.state.success,
                'errors': self.state.errors,
                'progress': (self.state.processed / self.state.total_images * 100)
            })
            
        except Exception as e:
            self.logger.error(f"Error actualizando progreso: {e}")
            
    def _add_result_to_table(self, result: Dict[str, Any]):
        """Agrega un resultado a la tabla."""
        try:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            
            # Archivo
            filename = Path(result['path']).name
            self.results_table.setItem(row, 0, QTableWidgetItem(filename))
            
            # Estado
            status = result['status'].title()
            status_item = QTableWidgetItem(status)
            status_item.setBackground(QBrush(self.RESULT_COLORS.get(
                status.lower(), QColor(255, 255, 255)
            )))
            self.results_table.setItem(row, 1, status_item)
            
            # Tiempo
            time_str = f"{result.get('processing_time', 0):.2f}s"
            self.results_table.setItem(row, 2, QTableWidgetItem(time_str))
            
            # Detalles
            details = self._format_result_details(result)
            self.results_table.setItem(row, 3, QTableWidgetItem(details))
            
            # Auto-scroll
            self.results_table.scrollToBottom()
            
        except Exception as e:
            self.logger.error(f"Error agregando resultado a tabla: {e}")
            
    def _format_result_details(self, result: Dict[str, Any]) -> str:
        """Formatea los detalles del resultado."""
        details = []
        
        if error := result.get('error'):
            return f"Error: {error}"
            
        if data := result.get('data', {}):
            if 'processed_fields' in data:
                details.append(
                    f"Campos: {data['processed_fields']}/{data['total_fields']}"
                )
            if data.get('aligned'):
                details.append("Alineada")
                
        if barcode := result.get('barcode'):
            details.append(f"Código: {barcode}")
            
        return " | ".join(details) if details else "--"
        
    def stop(self):
        """Detiene el monitoreo."""
        self.status_label.setText("Detenido")
        self.processing_stopped.emit()
        
    def complete(self):
        """Finaliza el monitoreo."""
        self.status_label.setText("Completado")
        
    def prepare_for_close(self):
        """Prepara el widget para el cierre."""
        self.stop()
        return True