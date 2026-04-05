"""
Configuración de la aplicación de análisis de tránsito peatonal.
"""
from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
MODELS_DIR = BASE_DIR / "models"

# Directorios de almacenamiento
UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"

# Modelo YOLO
YOLO_MODEL_PATH = MODELS_DIR / "yolo26m.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.25

# Clases objetivo (COCO dataset)
CLASSES = {
    0: "person",
}

# Configuración de video
MAX_VIDEO_SIZE_MB = 500
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# Configuración de procesamiento
FPS_PROCESS = 30  # FPS para procesar
TRACKING_MAX_AGE = 90  # Frames máximos sin detección antes de eliminar track (3 segundos a 30fps)

# Colores para visualización (BGR)
COLORS = {
    "person": (255, 0, 0),   # Azul
}

# Región de conteo custom (polígono de 4 vértices)
COUNT_REGION = {
    "vertices": [
        #[x, y]
        [445, 180],  # superior-izq
        [847, 180],  # superior-der

        [960, 540],  # inferior-der
        [360, 540]   # inferior-izq
    ],
    "name": "REGION"
}
