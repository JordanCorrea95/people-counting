"""
Módulo de servicios.
"""
from .people_detector import peopleDetector
from .people_processor import VideoProcessor

__all__ = ["peopleDetector", "VideoProcessor"]
