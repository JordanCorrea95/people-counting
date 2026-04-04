"""
Modelos de datos para detecciones y tracking.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class Direction(str, Enum):
    """Direcciones de movimiento."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    UNKNOWN = "unknown"


@dataclass
class peopleDetection:
    """Representa una detección."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[List] = None  # Puntos de segmentación
    centroid: Tuple[int, int] = field(default=(0, 0))
    visible_point: Tuple[int, int] = field(default=(0, 0))
    bottom_point: Tuple[int, int] = field(default=(0, 0))

    def __post_init__(self):
        """Calcula puntos derivados del bounding box si no se proporcionan."""
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            height = max(1, y2 - y1)
            if self.centroid == (0, 0):
                self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            if self.visible_point == (0, 0):
                # Punto visual a 3/4 de altura desde la base del bbox (más alto que el centro).
                self.visible_point = ((x1 + x2) // 2, y2 - int(0.75 * height))
            if self.bottom_point == (0, 0):
                self.bottom_point = ((x1 + x2) // 2, y2)


@dataclass
class peopleTrack:
    """Representa el seguimiento de un objeto a lo largo del tiempo."""
    track_id: int
    class_name: str
    positions: List[Tuple[int, int]] = field(default_factory=list)
    direction: Direction = Direction.UNKNOWN
    frames_since_last_detection: int = 0

    def update_position(self, position: Tuple[int, int], class_name: Optional[str] = None):
        """Actualiza la posición y el historial de clases."""
        self.positions.append(position)
        self.frames_since_last_detection = 0
        if class_name:
            self.class_name = class_name

        # Mantener solo las últimas N posiciones para calcular dirección
        if len(self.positions) >= 5:
            self.positions = self.positions[-5:]

        self._calculate_direction()

    def _calculate_direction(self):
        """Calcula la dirección de movimiento basado en el historial de N posiciones."""
        if len(self.positions) < 5:
            return

        # Comparar posición actual con posición hace N frames
        current_pos = self.positions[-1]
        past_pos = self.positions[0]

        # Cambio vertical
        dy = current_pos[1] - past_pos[1]

        # Determinar dirección predominante (sólo consideraremos eje y)
        if dy > 0:
            self.direction = Direction.SOUTH
        elif dy < 0:
            self.direction = Direction.NORTH
        else:
            # dy == 0 (No hay movimiento)
            self.direction = None

    def increment_missed_frames(self):
        self.frames_since_last_detection += 1


@dataclass
class peopleStats:
    """Estadísticas de tránsito peatonal."""
    total_people: int = 0
    by_class: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_lane: dict = field(default_factory=dict)

    def to_dict(self):
        """Convierte las estadísticas a diccionario."""
        return {
            "total_people": self.total_people,
            "by_class": self.by_class,
            "by_direction": self.by_direction,
            "by_lane": self.by_lane
        }
