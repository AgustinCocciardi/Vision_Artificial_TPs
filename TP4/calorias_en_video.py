import os
from ultralytics import YOLO
import cv2
import random

# --- Configuración de rutas ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best.pt")

# --- Umbrales ---
UMBRAL_CONF = 0.8
UMBRAL_IOA = 0.9

# --- Clases del dataset ---
CLASES = [
    "Banana", "Jackfruit", "Mango", "Litchi", "Hog Plum",
    "Papaya", "Grapes", "Apple", "Orange", "Guava"
]

# --- Calorías estimadas por unidad ---
CALORIAS_POR_FRUTA = {
    "Banana": 89, "Jackfruit": 155, "Mango": 60, "Litchi": 66, "Hog Plum": 75,
    "Papaya": 43, "Grapes": 69, "Apple": 52, "Orange": 47, "Guava": 68
}

# --- Colores únicos por clase ---
COLORS = {
    cls: tuple([int(x) for x in (
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    )]) for cls in CLASES
}

def get_color_for_label(label):
    for variant in [label, label.capitalize(), label.lower()]:
        if variant in COLORS:
            return COLORS[variant]
    h = abs(hash(label)) % (256**3)
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def ioa(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / box_area(boxA) if box_area(boxA) > 0 else 0.0

def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    union = box_area(boxA) + box_area(boxB) - inter
    return inter / union if union > 0 else 0.0

# --- NUEVO FILTRO: quedarse con la detección de mayor confianza si hay solapamiento ---
def filtrar_boxes(dets):
    dets = sorted(dets, key=lambda d: d['conf'], reverse=True)  # Ordenar por confianza
    keep = []

    for det in dets:
        solapa_con_algo = False
        for k in keep:
            # Solo comparar detecciones de la misma fruta
            if det['label'].lower() == k['label'].lower():
                if iou(det['xyxy'], k['xyxy']) >= UMBRAL_IOA:  # Si solapan lo suficiente
                    solapa_con_algo = True
                    break
        if not solapa_con_algo:
            keep.append(det)

    return keep

# --- Cargar modelo YOLO ---
model = YOLO(MODEL_PATH)

# --- Iniciar cámara ---
cap = cv2.VideoCapture(0)  # Cámara 0 confirmada

if not cap.isOpened():
    print(" No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    dets = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < UMBRAL_CONF:
                continue
            cls = int(box.cls[0])
            label = result.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            dets.append({"label": label, "conf": conf, "xyxy": (x1, y1, x2, y2)})

    kept = filtrar_boxes(dets)

    calorias_totales = 0

    for det in kept:
        x1, y1, x2, y2 = det["xyxy"]
        color_rgb = get_color_for_label(det["label"])
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

        calorias = CALORIAS_POR_FRUTA.get(det["label"], 0)
        calorias_totales += calorias

        cv2.putText(frame, f"{det['label']} {det['conf']*100:.1f}% - {calorias} kcal",
                    (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    cv2.putText(frame, f"TOTAL: {calorias_totales} kcal",
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Deteccion de Frutas - Calorias en Vivo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
