# People Inventory Counting (Region-Based)

API y pipeline de video para detectar, segmentar y contar (inventario) personas dentro de una region de conteo definida por un poligono de 4 vertices. Usa FastAPI, OpenCV y Ultralytics YOLO (segmentacion).

## Caracteristicas

- Deteccion y segmentacion de personas (clase COCO `person`) con YOLO-seg via Ultralytics.
- Tracking por `track_id` usando `model.track(..., persist=True)`.
- Conteo tipo inventario: cuenta cuantas personas estan dentro de la region en cada frame.
- Region de conteo configurable como poligono de 4 vertices en `app/config/settings.py`.
- Video de salida anotado: region (poligono), bbox, mascara (si existe), ID y estadisticas.

## Requisitos

- Python 3.10+
- (Opcional) GPU/CUDA para acelerar inferencia

## Instalacion

### 1. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Decargar modelo YOLO (.pt)

```bash
mkdir -p models
curl -L -o models/yolo26m-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
```

La aplicacion valida que el archivo del modelo:

- Exista y sea archivo
- Tenga extension `.pt`
- Tenga un tamaño mínimo (para detectar archivos corruptos/placeholder)

## Configuracion

Archivo: `app/config/settings.py`

- `YOLO_MODEL_PATH`, `YOLO_CONFIDENCE_THRESHOLD`, `YOLO_IOU_THRESHOLD`
- `TRACKING_MAX_AGE` (frames maximos sin deteccion antes de borrar un track)
- `COUNT_REGION.vertices` (poligono de 4 vertices)

Ejemplo de region:

```python
COUNT_REGION = {
    "vertices": [
        [445, 180],  # superior-izq
        [847, 180],  # superior-der
        [960, 540],  # inferior-der
        [360, 540],  # inferior-izq
    ],
    "name": "REGION"
}
```

## Como funciona el conteo

- El conteo se decide con el punto inferior del bounding box (`bottom_point`), no con el centro.
- El punto mostrado en pantalla (`visible_point`) esta a 3/4 de altura del bbox (solo visual).
- Se considera "dentro" si el punto de conteo cae dentro del poligono (`pointPolygonTest`).

## Ejecutar la API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Endpoints

- `GET /health`: valida presencia/tamano del modelo.
- `POST /api/upload`: sube un video para procesar.
- `GET /api/status/{video_id}`: progreso y estadisticas al finalizar.
- `GET /api/download/{video_id}`: descarga el video anotado.
- `DELETE /api/delete/{video_id}`: elimina archivos del video.

Ejemplo upload:

```bash
curl -X POST "http://localhost:8000/api/upload" -F "file=@mi_video.mp4"
```

## Estructura

```
people-counting/
├── app/
│   ├── config/
│   ├── models/
│   ├── services/
│   │   ├── people_detector.py
│   │   └── people_processor.py
│   ├── utils/
│   └── main.py
├── models/
├── storage/
│   ├── uploads/
│   └── processed/
└── requirements.txt
```

## Tecnologias

- FastAPI / Uvicorn
- Ultralytics YOLO (segmentacion)
- OpenCV / NumPy

## Licencia

MIT
