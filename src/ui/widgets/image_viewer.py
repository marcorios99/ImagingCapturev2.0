import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea,
    QDialog, QSizePolicy, QMenu, QHBoxLayout,
    QFrame, QMessageBox
)
from PySide6.QtCore import (
    Qt, Signal, QPoint, QPointF, QTimer
)
from PySide6.QtGui import (
    QImage, QPixmap, QTransform, QMouseEvent, QWheelEvent, 
    QImageReader
)

@dataclass
class ViewerState:
    """Estado del visor de imágenes."""
    zoom_factor: float = 1.0
    rotation_angle: float = 0.0
    is_flipped_h: bool = False
    is_flipped_v: bool = False
    fit_to_view: bool = True

class ImageViewerDialog(QDialog):
    """Diálogo modal para visualización de imágenes."""
    
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visor de Imagen")
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        self.resize(1024, 768)
        
        # Widget central
        self.viewer = ImageViewer(image_path)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Barra de estado
        self.status_bar = self._create_status_bar()
        
        # Agregar widgets
        layout.addWidget(self.viewer)
        layout.addWidget(self.status_bar)
        
        # Conectar señales
        self.viewer.mouse_moved.connect(self._update_cursor_pos)
        self.viewer.zoom_changed.connect(self._update_zoom)
        
        # Configuración inicial
        self.setWindowState(Qt.WindowMaximized)
        
    def _create_status_bar(self) -> QFrame:
        """Crea la barra de estado."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.NoFrame)
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4, 2, 4, 2)
        
        # Etiquetas de información
        self.pos_label = QLabel("Posición: --")
        self.zoom_label = QLabel("Zoom: 100%")
        self.size_label = QLabel("--")
        
        # Actualizar tamaño inicial
        if self.viewer.current_image:
            w = self.viewer.current_image.width()
            h = self.viewer.current_image.height()
            self.size_label.setText(f"{w} x {h} px")
        
        layout.addWidget(self.pos_label)
        layout.addStretch()
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.size_label)
        
        return frame
        
    def _update_cursor_pos(self, pos: QPointF):
        """Actualiza la posición del cursor."""
        self.pos_label.setText(f"Posición: {int(pos.x())}, {int(pos.y())}")
        
    def _update_zoom(self, zoom: float):
        """Actualiza el nivel de zoom."""
        self.zoom_label.setText(f"Zoom: {int(zoom * 100)}%")
        
    def show_info_dialog(self):
        """Muestra información de la imagen."""
        if not self.viewer.current_image or not self.viewer.image_path:
            return
            
        try:
            path = Path(self.viewer.image_path)
            img = self.viewer.current_image
            
            # Obtener DPI
            dpi_x = img.physicalDpiX()
            dpi_y = img.physicalDpiY()
            
            info = {
                "Nombre": path.name,
                "Dimensiones": f"{img.width()} x {img.height()} px",
                "Resolución": f"{dpi_x} x {dpi_y} DPI",
                "Formato": img.format(),
                "Profundidad": f"{img.depth()} bits",
                "Tamaño archivo": f"{path.stat().st_size / 1024:.1f} KB",
                "Modificado": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }
            
            # Crear mensaje formateado
            msg_text = ""
            for key, value in info.items():
                msg_text += f"<b>{key}:</b> {value}<br>"
            
            msg = QMessageBox(self)
            msg.setWindowTitle("Información de Imagen")
            msg.setText(msg_text)
            msg.setTextFormat(Qt.RichText)
            msg.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error obteniendo información: {str(e)}")
            
            
class ImageViewer(QWidget):
    """Widget de visualización de imágenes."""
    
    # Señales
    mouse_moved = Signal(QPointF)
    zoom_changed = Signal(float)
    
    # Configuración
    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0
    
    def __init__(self, image_path: Optional[str] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Estado
        self.state = ViewerState()
        self.image_path = None
        self.current_image: Optional[QImage] = None
        self.displayed_pixmap: Optional[QPixmap] = None
        self.last_mouse_pos = QPoint()
        self.panning = False
        
        # Configuración
        self.setup_ui()
        
        # Cargar imagen si se proporciona
        if image_path:
            self.load_image(image_path)
            
    def setup_ui(self):
        """Configura la interfaz del widget."""
        self.setMinimumSize(400, 300)
        
        # Layout principal
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # ScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameStyle(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Label para la imagen
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(1, 1)
        self.scroll_area.setWidget(self.image_label)
        
        layout.addWidget(self.scroll_area)
        
        # Configuración de eventos
        self.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
        
    def load_image(self, image_path: str) -> bool:
        """Carga una imagen desde el archivo especificado."""
        try:
            # Validar ruta
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"No se encuentra el archivo: {image_path}")
                
            # Verificar formato
            reader = QImageReader(str(path))
            if not reader.canRead():
                raise ValueError(f"Formato de imagen no soportado: {path.suffix}")
            
            # Cargar imagen
            reader.setAutoTransform(True)  # Preservar metadatos
            self.current_image = reader.read()
            
            if self.current_image.isNull():
                raise ValueError("No se pudo cargar la imagen")
            
            # Actualizar estado
            self.image_path = str(path)
            self.reset_state()
            
            # Mostrar imagen
            self.update_display()
            if self.state.fit_to_view:
                self.zoom_to_fit()
                
            # Actualizar información de tamaño
            if parent := self.parent():
                if hasattr(parent, 'size_label'):
                    w = self.current_image.width()
                    h = self.current_image.height()
                    parent.size_label.setText(f"{w} x {h} px")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando imagen: {e}")
            QMessageBox.critical(self, "Error", f"Error al cargar imagen: {str(e)}")
            return False
            
    def update_display(self):
        """Actualiza la visualización con las transformaciones actuales."""
        if not self.current_image:
            return
            
        try:
            # Crear copia para transformaciones
            transformed = self.current_image.copy()
            
            # Aplicar transformaciones
            transform = QTransform()
            
            # Rotación
            if self.state.rotation_angle != 0:
                transform.rotate(self.state.rotation_angle)
            
            # Volteo
            if self.state.is_flipped_h:
                transform.scale(-1, 1)
            if self.state.is_flipped_v:
                transform.scale(1, -1)
                
            transformed = transformed.transformed(transform, Qt.SmoothTransformation)
            
            # Aplicar zoom
            if self.state.zoom_factor != 1.0:
                scaled_size = transformed.size() * self.state.zoom_factor
                transformed = transformed.scaled(
                    scaled_size.width(),
                    scaled_size.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
 
            # Actualizar pixmap
            self.displayed_pixmap = QPixmap.fromImage(transformed)
            self.image_label.setPixmap(self.displayed_pixmap)
            
            # Emitir cambios
            self.zoom_changed.emit(self.state.zoom_factor)
            
        except Exception as e:
            self.logger.error(f"Error actualizando display: {e}")
                
    def zoom_to_fit(self):
        """Ajusta el zoom para que la imagen se ajuste a la ventana."""
        if not self.current_image or not self.scroll_area.viewport().size().isValid():
            return
            
        # Obtener dimensiones
        viewport_size = self.scroll_area.viewport().size()
        image_size = self.current_image.size()
        
        if self.state.rotation_angle in (90, 270):
            image_size.transpose()
        
        # Calcular factores
        width_ratio = viewport_size.width() / image_size.width()
        height_ratio = viewport_size.height() / image_size.height()
        
        # Usar el factor menor para que toda la imagen sea visible
        self.state.zoom_factor = min(width_ratio, height_ratio) * 0.95
        self.state.fit_to_view = True
        self.update_display()
            
    def reset_state(self):
        """Resetea el estado del visor."""
        self.state = ViewerState()
        
    def get_image_coordinates(self, pos: QPoint) -> QPointF:
        """Convierte coordenadas de pantalla a coordenadas de imagen."""
        if not self.displayed_pixmap:
            return QPointF()
            
        # Obtener posición relativa al label
        image_pos = self.image_label.mapFrom(self, pos)
        label_center = QPoint(
            self.image_label.width() // 2,
            self.image_label.height() // 2
        )
        
        # Ajustar por centrado
        image_x = (image_pos.x() - label_center.x()) / self.state.zoom_factor
        image_y = (image_pos.y() - label_center.y()) / self.state.zoom_factor
        
        # Ajustar por transformaciones
        transform = QTransform()
        
        if self.state.rotation_angle != 0:
            transform.rotate(-self.state.rotation_angle)
        if self.state.is_flipped_h:
            transform.scale(-1, 1)
        if self.state.is_flipped_v:
            transform.scale(1, -1)
            
        return transform.map(QPointF(image_x, image_y))
        
    def mousePressEvent(self, event: QMouseEvent):
        """Maneja el evento de presionar el mouse."""
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Maneja el evento de mover el mouse."""
        image_pos = self.get_image_coordinates(event.pos())
        self.mouse_moved.emit(image_pos)
        
        if self.panning:
            delta = event.pos() - self.last_mouse_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            self.last_mouse_pos = event.pos()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Maneja el evento de soltar el mouse."""
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            
    def wheelEvent(self, event: QWheelEvent):
        """Maneja el evento de la rueda del mouse."""
        # Zoom con la rueda
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        
        new_zoom = self.state.zoom_factor * factor
        if self.MIN_ZOOM <= new_zoom <= self.MAX_ZOOM:
            # Mantener el punto bajo el cursor
            old_pos = self.get_image_coordinates(event.position().toPoint())
            
            self.state.zoom_factor = new_zoom
            self.state.fit_to_view = False
            self.update_display()
            
            new_pos = self.get_image_coordinates(event.position().toPoint())
            
            # Ajustar scroll para mantener el punto
            delta_x = (new_pos.x() - old_pos.x()) * self.state.zoom_factor
            delta_y = (new_pos.y() - old_pos.y()) * self.state.zoom_factor
            
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() + int(delta_x)
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() + int(delta_y)
            )
            
    def resizeEvent(self, event):
        """Maneja el evento de redimensionar la ventana."""
        super().resizeEvent(event)
        if self.state.fit_to_view:
            self.zoom_to_fit()
            
    def contextMenuEvent(self, event):
        """Crea el menú contextual."""
        if not self.current_image:
            return
            
        menu = QMenu(self)
        # Acciones de transformación
        transform_menu = menu.addMenu("Transformar")
        transform_menu.addAction("Rotar 90° Derecha", lambda: self.rotate(90))
        transform_menu.addAction("Rotar 90° Izquierda", lambda: self.rotate(-90))
        transform_menu.addAction("Voltear Horizontal", self.flip_horizontal)
        transform_menu.addAction("Voltear Vertical", self.flip_vertical)
        
        menu.addSeparator()
        menu.addAction("Información", lambda: self.parent().show_info_dialog())
        
        menu.exec_(event.globalPos())
        
    def rotate(self, degrees: float):
        """Rota la imagen el número de grados especificado."""
        self.state.rotation_angle = (self.state.rotation_angle + degrees) % 360
        self.update_display()
        
    def flip_horizontal(self):
        """Voltea la imagen horizontalmente."""
        self.state.is_flipped_h = not self.state.is_flipped_h
        self.update_display()
        
    def flip_vertical(self):
        """Voltea la imagen verticalmente."""
        self.state.is_flipped_v = not self.state.is_flipped_v
        self.update_display()