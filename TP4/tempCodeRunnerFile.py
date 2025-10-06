import os
from ultralytics import YOLO

# Ruta absoluta al script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(SCRIPT_DIR, 'data.yaml')

# Cargar modelo preentrenado (puede ser 'yolov8s.pt', 'yolov8n.pt', etc.)
modelo = YOLO('yolov8s.pt')

# Entrenar
modelo.train(
    data=DATA_YAML,     # path al archivo YAML
    epochs=10,            # podés subir o bajar
    imgsz=640,            # tamaño de imagen
    batch=16,             # ajustar según tu GPU/CPU
    workers=0,            # evita errores en Windows
    device='cpu'          # 'cpu' si no está disponible el GPU
)

# Opcional: evaluación
modelo.val()
