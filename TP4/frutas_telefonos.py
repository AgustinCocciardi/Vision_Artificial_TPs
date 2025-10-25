import os
import cv2
import random
from ultralytics import YOLO

# --- Configuración de rutas ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FRUITS_MODEL_PATH = os.path.join(CURRENT_DIR, "best.pt")
PHONE_MODEL_PATH = os.path.join(CURRENT_DIR, "phone.pt")

# --- Umbrales ---
UMBRAL_CONF = 0.5
UMBRAL_IOU = 0.9

# --- Clases de frutas ---
CLASES_FRUTAS = [
    "Banana", "Jackfruit", "Mango", "Litchi", "Hog Plum",
    "Papaya", "Grapes", "Apple", "Orange", "Guava"
]

# --- Clase de teléfono ---
CLASES_PHONE = ["Phone"]

# --- Colores por clase ---
TODAS_CLASES = CLASES_FRUTAS + CLASES_PHONE
COLORS = {
    cls: tuple(random.randint(0, 255) for _ in range(3)) for cls in TODAS_CLASES
}

def get_color_for_label(label):
    return COLORS.get(label, (0, 255, 0))

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area(boxA) + box_area(boxB) - inter
    return inter / union if union > 0 else 0.0

def filtrar_boxes(dets):
    dets = sorted(dets, key=lambda d: d['conf'], reverse=True)
    keep = []
    for det in dets:
        solapa = False
        for k in keep:
            if det['label'].lower() == k['label'].lower():
                if iou(det['xyxy'], k['xyxy']) > UMBRAL_IOU:
                    solapa = True
                    break
        if not solapa:
            keep.append(det)
    return keep

# --- Cargar modelos ---
model_frutas = YOLO(FRUITS_MODEL_PATH)
model_phone = YOLO(PHONE_MODEL_PATH)

# --- Iniciar cámara ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Inferencia con ambos modelos ---
    results_frutas = model_frutas(frame)
    results_phone = model_phone(frame)

    dets = []

    # --- Detecciones de frutas ---
    for result in results_frutas:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < UMBRAL_CONF:
                continue
            cls = int(box.cls[0])
            label = result.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            dets.append({"label": label, "conf": conf, "xyxy": (x1, y1, x2, y2)})

    # --- Detecciones de teléfono ---
    for result in results_phone:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < UMBRAL_CONF:
                continue
            cls = int(box.cls[0])
            label = result.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            dets.append({"label": label, "conf": conf, "xyxy": (x1, y1, x2, y2)})

    # --- Filtrar solapamientos ---
    kept = filtrar_boxes(dets)

    # --- Dibujar detecciones ---
    for det in kept:
        x1, y1, x2, y2 = det["xyxy"]
        color = get_color_for_label(det["label"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det['label']} {det['conf']*100:.1f}%",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detección de Frutas y Teléfonos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
