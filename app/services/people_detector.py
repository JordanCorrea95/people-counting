"""
Servicio de detección, seguimiento y conteo de inventario de personas.
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from ultralytics import YOLO

from app.config import settings
from app.models import peopleDetection, peopleTrack, peopleStats

logger = logging.getLogger(__name__)


class peopleDetector:
    """Detector de personas usando YOLO con segmentación."""

    @staticmethod
    def _validate_model_file(model_path: Path):
        """Valida que el archivo del modelo sea utilizable antes de cargarlo."""
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        if not model_path.is_file():
            raise ValueError(f"La ruta del modelo no es un archivo: {model_path}")
        if model_path.suffix.lower() != ".pt":
            raise ValueError(f"Formato de modelo no válido (se esperaba .pt): {model_path}")
        if model_path.stat().st_size < 1024:
            raise ValueError(f"Archivo de modelo inválido o corrupto: {model_path}")

    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa el detector de personas.

        Args:
            model_path: Ruta al modelo YOLO. Si es None, usa el de configuración.
        """
        self.model_path = model_path or settings.YOLO_MODEL_PATH

        self._validate_model_file(self.model_path)

        # Cargar modelo YOLO
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar el modelo YOLO: {self.model_path}") from exc

        # Diccionario de tracks activos
        self.tracks: Dict[int, peopleTrack] = {}

        # Estadísticas
        self.stats = peopleStats()

        # Contador de IDs únicos
        self._next_track_id = 0

    def detect(
            self,
            frame: np.ndarray,
            conf_threshold: float = None,
            iou_threshold: float = None
    ) -> List[peopleDetection]:
        """
        Detecta personas en un frame.

        Args:
            frame: Frame de video
            conf_threshold: Umbral de confianza
            iou_threshold: Umbral de IoU

        Returns:
            Lista de detecciones de personas
        """
        conf = conf_threshold or settings.YOLO_CONFIDENCE_THRESHOLD
        iou = iou_threshold or settings.YOLO_IOU_THRESHOLD

        # Ejecutar detección con tracking (solo clase persona)
        results = self.model.track(
            frame,
            conf=conf,
            iou=iou,
            persist=True,
            classes=list(settings.CLASSES.keys()),
            verbose=False
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Obtener boxes y IDs de tracking
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    # Extraer información de la detección
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else self._next_track_id
                    self._next_track_id = max(self._next_track_id, track_id + 1)

                    # Obtener nombre de clase
                    class_name = settings.CLASSES.get(cls, "unknown")

                    # Crear detección
                    detection = peopleDetection(
                        track_id=track_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        mask=None
                    )

                    detections.append(detection)

        return detections

    def update_tracks(self, detections: List[peopleDetection]):
        """
        Actualiza los tracks con las nuevas detecciones.

        Args:
            detections: Lista de detecciones
        """
        detected_ids = set()

        for detection in detections:
            detected_ids.add(detection.track_id)

            if detection.track_id not in self.tracks:
                # Crear nuevo track
                self.tracks[detection.track_id] = peopleTrack(
                    track_id=detection.track_id,
                    class_name=detection.class_name
                )

            # Para conteo, usar el límite inferior del bbox (punto de apoyo en el suelo)
            self.tracks[detection.track_id].update_position(
                detection.bottom_point,
                class_name=detection.class_name
            )

        # Incrementar contador de frames perdidos para tracks no detectados
        for track_id in list(self.tracks.keys()):
            if track_id not in detected_ids:
                self.tracks[track_id].increment_missed_frames()

                # Eliminar tracks que llevan mucho tiempo sin detección
                if self.tracks[track_id].frames_since_last_detection > settings.TRACKING_MAX_AGE:
                    del self.tracks[track_id]

    @staticmethod
    def _is_inside_region(point: tuple[int, int], region: dict) -> bool:
        """
        Verifica si un punto está dentro del polígono de conteo.
        """
        polygon = np.array(region["vertices"], dtype=np.float32)
        pt = (float(point[0]), float(point[1]))
        return cv2.pointPolygonTest(polygon, pt, False) >= 0

    def update_inventory_stats(self, region: dict) -> Set[int]:
        """
        Actualiza inventario: cuántas personas están dentro de la región en el frame actual.
        """
        inside_track_ids: Set[int] = set()

        for track_id, track in self.tracks.items():
            if not track.positions:
                continue
            # Evita contar tracks "fantasma" cuando no hubo deteccion en el frame actual.
            if track.frames_since_last_detection != 0:
                continue
            current_pos = track.positions[-1]
            if self._is_inside_region(current_pos, region):
                inside_track_ids.add(track_id)

        total_inside = len(inside_track_ids)
        self.stats.total_people = total_inside
        self.stats.by_class = {"person": total_inside} if total_inside else {}
        self.stats.by_direction = {}
        self.stats.by_lane = {}

        if "by_class_direction" in self.stats.__dict__:
            del self.stats.__dict__["by_class_direction"]

        return inside_track_ids

    def get_stats(self) -> peopleStats:
        """Retorna las estadísticas actuales."""
        return self.stats

    def reset_stats(self):
        """Reinicia las estadísticas."""
        self.stats = peopleStats()
        self.tracks = {}
