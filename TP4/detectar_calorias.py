import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración de rutas ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best.pt")
TEST_IMAGES_DIR = os.path.join(CURRENT_DIR, "test_images")

# --- Calorías aproximadas por alimento ---
calorias_por_alimento = {
    "banana": 89,
    "bacon": 541,
    "bread": 265,
    "broccoli": 55,
    "butter": 717,
    "carrots": 41,
    "cheese": 402,
    "chicken": 239,
    "cucumber": 16,
    "eggs": 155,
    "fish": 206,
    "lettuce": 15,
    "milk": 42,
    "onions": 40,
    "peppers": 31,
    "potatoes": 77,
    "sausages": 301,
    "spinach": 23,
    "tomato": 18,
    "yogurt": 59
}

# --- Cargar modelo YOLO ---
model = YOLO(MODEL_PATH)

# --- Obtener lista de imágenes ---
image_files = [f for f in os.listdir(TEST_IMAGES_DIR)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()

# Variables globales para navegación
current_index = 0
fig, ax = plt.subplots(figsize=(8, 6))


def mostrar_imagen(index):
    """Muestra la imagen con bounding boxes y datos fuera de la imagen."""
    ax.clear()
    filename = image_files[index]
    image_path = os.path.join(TEST_IMAGES_DIR, filename)

    # Inferencia
    results = model(image_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detecciones = []
    conteo_detecciones = {}

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Guardar todas las detecciones
            detecciones.append({
                "label": label,
                "conf": conf,
                "xyxy": (x1, y1, x2, y2)
            })

            # Contar por clase
            conteo_detecciones[label] = conteo_detecciones.get(label, 0) + 1

    # --- Dibujar bounding boxes ---
    for det in detecciones:
        x1, y1, x2, y2 = det["xyxy"]
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- Calcular calorías totales ---
    calorias_totales = 0
    lineas_info = []

    for det in detecciones:
        label = det["label"]
        conf = det["conf"]
        calorias_unitarias = calorias_por_alimento.get(label, 0)
        lineas_info.append(f"{label}: conf={conf:.2f}, {calorias_unitarias} kcal/unidad")

    # Agrupado por clase
    for label, cantidad in conteo_detecciones.items():
        calorias_label = calorias_por_alimento.get(label, 0) * cantidad
        calorias_totales += calorias_label
        lineas_info.append(f"→ {label}: {cantidad} × {calorias_por_alimento.get(label, 0)} = {calorias_label} kcal")

    lineas_info.append(f"\nTOTAL ESTIMADO: {calorias_totales} kcal")

    # --- Mostrar imagen y texto ---
    ax.imshow(image_rgb)
    ax.axis("off")

    texto = "\n".join(lineas_info)
    plt.text(1.05, 0.5, texto, transform=ax.transAxes,
             fontsize=11, va='center', bbox=dict(facecolor='white', alpha=0.9))
    plt.title(f"{filename}  ({index+1}/{len(image_files)})")
    plt.tight_layout()
    fig.canvas.draw_idle()


def on_key(event):
    """Cambia de imagen con las flechas del teclado."""
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
