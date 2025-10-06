import os
import cv2
import shutil

# Obtiene la ruta absoluta del script y arma la ruta al dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DATASET = os.path.join(SCRIPT_DIR, "dataset")
YOLO_DATASET = os.path.join(SCRIPT_DIR, "dataset_yolo")

CLASS_MAP = {
    "bacon": 0,
    "banana": 1,
    "bread": 2,
    "broccoli": 3,
    "butter": 4,
    "carrots": 5,
    "cheese": 6,
    "chicken": 7,
    "cucumber": 8,
    "eggs": 9,
    "fish": 10,
    "lettuce": 11,
    "milk": 12,
    "onions": 13,
    "peppers": 14,
    "potatoes": 15,
    "sausages": 16,
    "spinach": 17,
    "tomato": 18,
    "yogurt": 19
}

SETS = ["train", "val", "test"]

def crear_dataset_yolo():
    if not os.path.exists(ORIGINAL_DATASET):
        print(f"❌ ERROR: No se encontró la carpeta: {ORIGINAL_DATASET}")
        return

    for subset in SETS:
        original_subset = os.path.join(ORIGINAL_DATASET, subset)

        if not os.path.exists(original_subset):
            print(f"⚠️ No se encontró la carpeta: {original_subset}")
            continue

        new_images_dir = os.path.join(YOLO_DATASET, subset, "images")
        new_labels_dir = os.path.join(YOLO_DATASET, subset, "labels")

        os.makedirs(new_images_dir, exist_ok=True)
        os.makedirs(new_labels_dir, exist_ok=True)

        print(f"\nProcesando: {subset}")

        for class_name in os.listdir(original_subset):
            class_path = os.path.join(original_subset, class_name)

            if not os.path.isdir(class_path):
                continue

            if class_name not in CLASS_MAP:
                print(f"⚠️ Clase '{class_name}' no está en CLASS_MAP. Se omite.")
                continue

            class_id = CLASS_MAP[class_name]

            for file in os.listdir(class_path):
                if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"⚠️ No se pudo leer: {img_path}")
                    continue

                new_img_path = os.path.join(new_images_dir, file)
                shutil.copy2(img_path, new_img_path)

                label_file = file.rsplit('.', 1)[0] + ".txt"
                label_path = os.path.join(new_labels_dir, label_file)

                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

        print(f"✅ Subset {subset} completado.")

if __name__ == "__main__":
    crear_dataset_yolo()
    print("\n✅ Dataset YOLO generado en carpeta 'dataset_yolo'")
