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
UMBRAL_CONF = 0.35
UMBRAL_IOA = 0.8

# --- Clases del dataset ---
CLASES = [
    "Banana", "Jackfruit", "Mango", "Litchi", "Hog Plum",
    "Papaya", "Grapes", "Apple", "Orange", "Guava"
]

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

# --- NUEVA FUNCIÓN: color por DETECCIÓN (determinístico por label+coords) ---
def get_color_for_det(det):
    """
    Genera un color entero (r,g,b) determinístico a partir de label+coords de la detección.
    Así cada bounding box tiene su propio color y se mantiene constante en cada ejecución.
    """
    s = f"{det['label']}_{det['xyxy'][0]}_{det['xyxy'][1]}_{det['xyxy'][2]}_{det['xyxy'][3]}"
    h = abs(hash(s)) % (256**3)
    r = (h >> 16) & 255
    g = (h >> 8) & 255
    b = h & 255
    return (r, g, b)

# --- Cargar modelo YOLO ---
model = YOLO(MODEL_PATH)

# --- Obtener imágenes ---
image_files = sorted([
    f for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

current_index = 0
fig, ax = plt.subplots(figsize=(10, 7))

# --- Utilidades ---
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

def filtrar_boxes(dets):
    n = len(dets)
    keep = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A, B = dets[i], dets[j]
            if A['label'].lower() != B['label'].lower():
                continue
            boxA, boxB = A['xyxy'], B['xyxy']
            if box_area(boxA) < box_area(boxB) and ioa(boxA, boxB) >= UMBRAL_IOA:
                if A['conf'] < B['conf']:
                    keep[i] = False
    return [d for k, d in enumerate(dets) if keep[k]]

# --- Mostrar imagen ---
def mostrar_imagen(index):
    ax.clear()
    filename = image_files[index]
    path = os.path.join(TEST_IMAGES_DIR, filename)

    results = model(path)
    # Leemos imagen en BGR (cv2) para dibujar con colores enteros correctamente
    img = cv2.imread(path)

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

    # --- Dibujar bounding boxes (sin texto dentro de la imagen) ---
    # Dibujamos sobre la imagen BGR con colores enteros y luego convertimos a RGB para mostrar
    for det in kept:
        x1, y1, x2, y2 = det["xyxy"]
        # color por detección (r,g,b)
        color_rgb = get_color_for_det(det)
        # cv2 usa BGR como enteros 0-255
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

    # Convertir BGR->RGB para matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mostrar imagen
    ax.imshow(img_rgb)
    ax.axis("off")

    # --- Texto lateral con colores (misma paleta que cada bbox) ---
    # Mostramos label y confianza con el mismo color de su bbox
    for i, det in enumerate(kept):
        color_rgb = get_color_for_det(det)
        # matplotlib espera color en 0..1 floats
        color_norm = (color_rgb[0]/255.0, color_rgb[1]/255.0, color_rgb[2]/255.0)
        ax.text(
            1.05, 0.9 - i * 0.05,
            f"{det['label']} ({det['conf']*100:.1f}%)",
            transform=ax.transAxes,
            fontsize=10,
            color=color_norm,
            va='top'
        )

    plt.title(f"{filename} ({index+1}/{len(image_files)})")
    plt.tight_layout()
    fig.canvas.draw_idle()

# --- Navegación ---
def on_key(event):
    global current_index
    if event.key == "right":
        current_index = (current_index + 1) % len(image_files)
        mostrar_imagen(current_index)
    elif event.key == "left":
        current_index = (current_index - 1) % len(image_files)
        mostrar_imagen(current_index)

fig.canvas.mpl_connect("key_press_event", on_key)
mostrar_imagen(current_index)
plt.show()
