"""
Servicio de procesamiento de video para análisis de tránsito peatonal.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import logging

from app.config import settings
from app.services.people_detector import peopleDetector
from app.utils import (
    get_video_info,
    create_video_writer,
    draw_text_with_background
)

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Procesador de video para detección y análisis de tránsito peatonal."""

    def __init__(self, detector: Optional[peopleDetector] = None):
        """
        Inicializa el procesador de video.

        Args:
            detector: Detector de personas. Si es None, crea uno nuevo.
        """
        self.detector = detector or peopleDetector()

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        model_path: Path = None,
        region_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> dict:
        """
        Procesa un video completo detectando y contando personas.

        Args:
            input_path: Ruta del video de entrada
            output_path: Ruta del video de salida
            model_path: Ruta del modelo YOLO a usar
            region_config: Configuración de región de conteo (usa default si es None)
            progress_callback: Función de callback para reportar progreso

        Returns:
            Diccionario con estadísticas del procesamiento
        """
        logger.info(f"Iniciando procesamiento de video: {input_path}")

        # Validar que existe el video
        if not input_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {input_path}")

        # Validar que exista el modelo
        model_path = model_path or settings.YOLO_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        logger.info(f"Modelo cargado: {model_path}")

        # Usar configuración por defecto si no se proporciona
        region_config = region_config or settings.COUNT_REGION

        # Resetear estadísticas
        self.detector.reset_stats()

        # Obtener información del video
        video_info = get_video_info(input_path)
        logger.info(f"Info del video: {video_info}")
        region_config = self._resolve_count_region(
            region_config,
            video_info["width"],
            video_info["height"]
        )

        # Abrir video de entrada
        cap = cv2.VideoCapture(str(input_path))

        # Crear video de salida
        writer = create_video_writer(
            output_path,
            video_info["width"],
            video_info["height"],
            video_info["fps"]
        )

        frame_count = 0
        total_frames = video_info["frame_count"]

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Detectar personas en el frame
                detections = self.detector.detect(frame)

                # Actualizar tracks
                self.detector.update_tracks(detections)

                # Conteo de inventario: personas dentro de la región en este frame
                inside_track_ids = self.detector.update_inventory_stats(region_config)

                # Dibujar visualizaciones en el frame
                annotated_frame = self._annotate_frame(
                    frame,
                    detections,
                    region_config,
                    inside_track_ids
                )

                # Escribir frame procesado
                writer.write(annotated_frame)

                # Callback de progreso
                if progress_callback and frame_count % 30 == 0:
                    progress_callback(frame_count, total_frames)

                # Log de progreso cada 10%
                if frame_count % (total_frames // 10 or 1) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progreso: {progress:.1f}% ({frame_count}/{total_frames})")

        finally:
            # Liberar recursos
            cap.release()
            writer.release()

        logger.info(f"Procesamiento completado: {frame_count} frames procesados")

        # Retornar estadísticas
        stats = self.detector.get_stats()
        return {
            "video_info": video_info,
            "frames_processed": frame_count,
            "statistics": stats.to_dict(),
            "output_path": str(output_path)
        }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list,
        region_config: dict,
        inside_track_ids: set[int]
    ) -> np.ndarray:
        """
        Anota el frame con detecciones, segmentaciones y estadísticas.

        Args:
            frame: Frame original
            detections: Lista de detecciones
            region_config: Configuración de región poligonal
            inside_track_ids: IDs de tracks dentro de la región

        Returns:
            Frame anotado
        """
        annotated = frame.copy()

        # Dibujar región poligonal de conteo
        polygon = np.array(region_config["vertices"], dtype=np.int32)
        cv2.polylines(annotated, [polygon], isClosed=True, color=(255, 0, 255), thickness=2)
        label_x = int(polygon[:, 0].min()) + 2
        label_y = max(20, int(polygon[:, 1].max()) + 12)
        draw_text_with_background(
            annotated,
            region_config.get("name", "REGION"),
            (label_x, label_y),
            font_scale=0.45,
            thickness=1,
            text_color=(255, 255, 255),
            bg_color=(255, 0, 255),
            padding=3
        )

        # Dibujar detecciones
        for detection in detections:
            is_inside = detection.track_id in inside_track_ids
            color = (0, 200, 0) if is_inside else settings.COLORS.get(detection.class_name, (0, 255, 255))

            # Dibujar máscara de segmentación si existe
            if detection.mask and len(detection.mask) > 0:
                mask_overlay = annotated.copy()
                points = np.array(detection.mask, dtype=np.int32)
                cv2.fillPoly(mask_overlay, [points], color)
                # Aplicar overlay con transparencia
                cv2.addWeighted(mask_overlay, 0.3, annotated, 0.7, 0, annotated)

                # Dibujar contorno de la máscara
                cv2.polylines(annotated, [points], True, color, thickness=1)

            # Dibujar bounding box
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=1)

            # Preparar etiqueta
            label = f"{detection.class_name.upper()} {detection.confidence:.2f}"

            # Dibujar etiqueta de clase con fondo
            draw_text_with_background(
                annotated,
                label,
                (x1, y1 - 10),
                font_scale=0.4,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=color,
                padding=3
            )

            # Dibujar ID de tracking
            track_label = f"ID:{detection.track_id}"
            draw_text_with_background(
                annotated,
                track_label,
                (x1, y2 + 15),
                font_scale=0.3,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=color,
                padding=2
            )

            # Dibujar punto visual (3/4 del bbox hacia arriba)
            if is_inside:
                cv2.circle(annotated, detection.visible_point, 6, (0, 255, 0), 2)
                cv2.circle(annotated, detection.visible_point, 3, (255, 255, 255), -1)
            else:
                cv2.circle(annotated, detection.visible_point, 4, color, -1)

        # Dibujar estadísticas en el frame
        self._draw_statistics(annotated)

        return annotated

    def _resolve_count_region(self, region_config: dict, frame_width: int, frame_height: int) -> dict:
        """
        Normaliza la región poligonal de conteo y la restringe al tamaño del frame.
        """
        default_vertices = [
            [int(frame_width * 0.25), int(frame_height * 0.25)],
            [int(frame_width * 0.75), int(frame_height * 0.25)],
            [int(frame_width * 0.75), int(frame_height * 0.75)],
            [int(frame_width * 0.25), int(frame_height * 0.75)],
        ]
        vertices = region_config.get("vertices", default_vertices)

        if not isinstance(vertices, list) or len(vertices) != 4:
            raise ValueError("COUNT_REGION.vertices debe contener exactamente 4 vértices")

        normalized_vertices = []
        for vertex in vertices:
            if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
                raise ValueError("Cada vértice de COUNT_REGION.vertices debe tener forma [x, y]")
            try:
                x = int(vertex[0])
                y = int(vertex[1])
            except (TypeError, ValueError) as exc:
                raise ValueError("Las coordenadas de COUNT_REGION.vertices deben ser numéricas") from exc
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            normalized_vertices.append([x, y])

        return {
            "vertices": normalized_vertices,
            "name": region_config.get("name", "REGION_CONTEO")
        }

    def _draw_statistics(self, frame: np.ndarray):
        """
        Dibuja las estadísticas actuales en el frame.

        Args:
            frame: Frame a anotar
        """
        stats = self.detector.get_stats()

        # Posición inicial para las estadísticas
        x, y = 10, 30
        line_height = 25

        # Calcular altura del fondo según contenido
        num_lines = 1

        bg_height = num_lines * line_height + 20

        # Fondo semi-transparente para las estadísticas
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Total de personas
        draw_text_with_background(
            frame,
            f"Total Personas: {stats.total_people}",
            (x, y),
            font_scale=0.6,
            thickness=2,
            text_color=(0, 255, 0),
            bg_color=(0, 0, 0),
            padding=2
        )
