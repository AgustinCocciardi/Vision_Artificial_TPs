import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import random

# --- Configuración de rutas ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best.pt")
TEST_IMAGES_DIR = os.path.join(CURRENT_DIR, "test_images")

# --- Umbrales ---
#UMBRAL_CONF = 0.6      # Confianza mínima
#UMBRAL_IOA = 1       # Porcentaje de solapamiento para descartar box pequeño
UMBRAL_CONF = 0.01      # Confianza mínima
UMBRAL_IOA = 0.95       # Porcentaje de solapamiento para descartar box pequeño

# --- Clases del dataset ---
CLASES = [
    "Banana",
    "Jackfruit",
    "Mango",
    "Litchi",
    "Hog Plum",
    "Papaya",
    "Grapes",
    "Apple",
    "Orange",
    "Guava"
]

# --- Generar un color único para cada clase ---
COLORS = {cls: tuple([random.randint(0, 255) for _ in range(3)]) for cls in CLASES}

# --- Cargar modelo YOLO ---
model = YOLO(MODEL_PATH)

# --- Obtener lista de imágenes ---
image_files = [f for f in os.listdir(TEST_IMAGES_DIR)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()

# Variables globales para navegación
current_index = 0
fig, ax = plt.subplots(figsize=(8, 6))


# --- Funciones de filtrado ---
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def ioa(boxA, boxB):
    """Porcentaje del boxA que está dentro de boxB"""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    return inter_area / box_area(boxA) if box_area(boxA) > 0 else 0

def filtrar_boxes(detecciones):
    """Filtra boxes según el umbral de solapamiento"""
    filtradas = []
    for i, detA in enumerate(detecciones):
        superpuesto = False
        for j, detB in enumerate(detecciones):
            if i == j:
                continue
            if ioa(detA['xyxy'], detB['xyxy']) >= UMBRAL_IOA:
                if box_area(detB['xyxy']) >= box_area(detA['xyxy']):
                    superpuesto = True
                    break
        if not superpuesto:
            filtradas.append(detA)
    return filtradas


# --- Función para mostrar imagen ---
def mostrar_imagen(index):
    ax.clear()
    filename = image_files[index]
    image_path = os.path.join(TEST_IMAGES_DIR, filename)

    # Inferencia
    results = model(image_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detecciones = []

    # --- Obtener todas las detecciones primero ---
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            detecciones.append({
                "label": label,
                "conf": conf,
                "xyxy": (x1, y1, x2, y2)
            })

    # --- Revisar si hay algún solapamiento entre boxes de la misma clase ---
    existe_solapamiento = False
    for i, detA in enumerate(detecciones):
        for j, detB in enumerate(detecciones):
            if i == j or detA['label'] != detB['label']:
                continue
            if ioa(detA['xyxy'], detB['xyxy']) > 0:
                existe_solapamiento = True
                break
        if existe_solapamiento:
            break

    # --- Filtrado según existencia de solapamiento ---
    if existe_solapamiento:
        # Aplicar umbral de confianza
        detecciones = [d for d in detecciones if d['conf'] >= UMBRAL_CONF]
        # Filtrar por solapamiento
        detecciones = filtrar_boxes(detecciones)
    # Si no hay solapamiento, dejamos todas las detecciones

    # --- Dibujar bounding boxes con color según clase ---
    for det in detecciones:
        x1, y1, x2, y2 = det["xyxy"]
        label = det["label"]
        color = COLORS[label]
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)

    # --- Generar texto con las detecciones ---
    lineas_info = [f"{det['label']}: conf={det['conf']:.2f}" for det in detecciones]

    # --- Mostrar imagen y texto ---
    ax.imshow(image_rgb)
    ax.axis("off")
    texto = "\n".join(lineas_info) if lineas_info else "Sin detecciones"
    plt.text(1.05, 0.5, texto, transform=ax.transAxes,
             fontsize=11, va='center', bbox=dict(facecolor='white', alpha=0.9))
    plt.title(f"{filename}  ({index+1}/{len(image_files)})")
    plt.tight_layout()
    fig.canvas.draw_idle()


# --- Función para navegación con teclas ---
def on_key(event):
    global current_index
    if event.key == "right":
        current_index = (current_index + 1) % len(image_files)
        mostrar_imagen(current_index)
    elif event.key == "left":
        current_index = (current_index - 1) % len(image_files)
        mostrar_imagen(current_index)


# --- Conectar eventos ---
fig.canvas.mpl_connect("key_press_event", on_key)

# --- Mostrar la primera imagen ---
mostrar_imagen(current_index)
plt.show()
