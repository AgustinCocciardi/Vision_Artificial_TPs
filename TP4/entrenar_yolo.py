from ultralytics import YOLO

# Cargar modelo preentrenado (puede ser 'yolov8s.pt', 'yolov8n.pt', etc.)
modelo = YOLO('yolov8s.pt')

# Entrenar
modelo.train(
    data='data.yaml',     # path al archivo YAML
    epochs=50,            # podés subir o bajar
    imgsz=640,            # tamaño de imagen
    batch=16,             # ajustar según tu GPU/CPU
    workers=0,            # evita errores en Windows
    device=0              # 0 = GPU si tenés, 'cpu' si no
)

# Opcional: evaluación
modelo.val()
