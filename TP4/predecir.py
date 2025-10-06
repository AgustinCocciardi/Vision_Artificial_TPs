import os
import cv2
from ultralytics import YOLO

# Ruta del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta al modelo entrenado
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

# Cargar modelo YOLOv8
modelo = YOLO(MODEL_PATH)

# Lista de imágenes a probar
imagenes = [
    "banana.44.jpg", "bacon.306.jpg", "bread.156.jpg", "brocolo.71.jpg",
    "butter.121.jpg", "carrots.119.jpg", "cheese.66.jpg", "chicken.5.jpg",
    "cucumber.124.jpg", "eggs.168.jpg", "fish.40.jpg", "lettuce.191.jpg",
    "milk.102.jpg", "onions.18.jpg", "peppers.44.jpg", "potatoes.22.jpg",
    "sausages.115.jpg", "spinach.49.jpg", "totamo.153.jpg", "yogurt.71.jpg"
]

# Carpeta donde están las imágenes (ajustar si las tenés en otra parte)
IMG_DIR = os.path.join(SCRIPT_DIR, "dataset_yolo", "val", "images")

# Recorrer cada imagen
for nombre_img in imagenes:
    ruta_img = os.path.join(IMG_DIR, nombre_img)

    if not os.path.exists(ruta_img):
        print(f"⚠️ No se encontró: {ruta_img}")
        continue

    # Ejecutar predicción
    resultados = modelo.predict(
        source=ruta_img,
        conf=0.25,
        save=False,
        show=False
    )

    r = resultados[0]
    img = r.plot()

    # Obtener etiquetas predichas
    clases_pred = [r.names[int(c)] for c in r.boxes.cls]
    confianzas = [float(conf) for conf in r.boxes.conf]

    # Mostrar información
    print(f"\n📸 Imagen: {nombre_img}")
    if clases_pred:
        for cls, conf in zip(clases_pred, confianzas):
            print(f"  🏷️ Clase: {cls} | 🎯 Confianza: {conf:.2f}")
    else:
        print("  ❌ No se detectó ninguna clase")

    # Mostrar imagen con detección
    cv2.imshow("Resultado YOLOv8", img)
    print("Presioná cualquier tecla para continuar a la siguiente imagen...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
