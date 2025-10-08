import os
import re

# === CONFIGURACIÓN ===
DATASET_PATH = r"C:\Users\Agustin Cocciardi\Documents\GitHub\Vision_Artificial_TPs\TP4\dataset_yolo"

# Mapeo entre nombre y clase
CLASSES = {
    "banana": 0,
    "jackfruit": 1,
    "mango": 2,
    "litchi": 3,
    "lichi": 3,           # variación común
    "hogplum": 4,
    "hog plum": 4,        # con espacio
    "papaya": 5,
    "grapes": 6,
    "apple": 7,
    "orange": 8,
    "guava": 9
}

def detectar_clase(nombre_archivo):
    """Detecta la clase a partir del nombre del archivo."""
    name = nombre_archivo.lower()
    for fruit, idx in CLASSES.items():
        if fruit in name.replace(" ", ""):
            return idx
    return None

def corregir_labels(carpeta):
    """Reescribe los .txt con la clase correcta según el nombre del archivo."""
    for root, _, files in os.walk(carpeta):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                class_id = detectar_clase(base_name)

                if class_id is None:
                    print(f"⚠️  No se detectó clase para {file}, se omite.")
                    continue

                # Leer etiquetas originales
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Reemplazar solo el primer número (ID de clase)
                nuevas = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    partes = re.split(r"\s+", line)
                    partes[0] = str(class_id)
                    nuevas.append(" ".join(partes))

                # Guardar nuevamente
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(nuevas))

                print(f"✅ Etiquetas corregidas para {file} -> clase {class_id}")

# Ejecutar
corregir_labels(DATASET_PATH)
print("✨ Corrección de etiquetas completada.")
