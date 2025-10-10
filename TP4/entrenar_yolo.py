import os
from ultralytics import YOLO

# --- Rutas absolutas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(SCRIPT_DIR, 'data.yaml')

# --- Cargar modelo base (preentrenado y liviano) ---
model = YOLO('yolov8s.pt')  # 'n' = nano (más rápido en CPU)

# --- Entrenamiento ---
model.train(
    data=DATA_YAML,       # ruta al archivo YAML
    epochs=50,            # 50 es buen punto de partida en CPU
    imgsz=512,            # tamaño más liviano
    batch=2,              # ideal para CPU
    workers=0,            # evita errores en Windows
    device='cpu',         # fuerza CPU
    name='fruits_cpu',    # nombre del experimento
    project='runs/train', # carpeta donde se guardan los resultados
    freeze=10,            # congela primeras capas (transfer learning)
    patience=10,          # early stopping
    augment=True          # aumenta diversidad del dataset
)

# --- Evaluación final ---
model.val(data=DATA_YAML)

print("\nEntrenamiento completado (modo CPU).")
print("Revisa la carpeta 'runs/train/fruits_cpu' para ver los resultados.")
