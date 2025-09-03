import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Nombre del archivo CSV con tu dataset
DATASET_FILE = "dataset.csv"

# Nombre del modelo entrenado
MODEL_FILE = "modelo_figuras.pkl"

def main():
    # Obtener ruta absoluta del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, DATASET_FILE)

    # Cargar dataset
    df = pd.read_csv(dataset_path)

    # Features (invariantes de Hu)
    X = df[["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7"]]

    # Etiquetas
    y = df["etiqueta"]

    # Crear modelo Decision Tree
    clf = DecisionTreeClassifier(
        criterion="gini",   # o "entropy"
        max_depth=None,     # podés ajustar para evitar overfitting
        random_state=42
    )

    # Entrenar
    clf.fit(X, y)

    # Guardar el modelo en la misma ruta del script
    ruta_guardado = os.path.join(script_dir, MODEL_FILE)
    joblib.dump(clf, ruta_guardado)

    print(f"✅ Modelo entrenado y guardado en: {ruta_guardado}")

if __name__ == "__main__":
    main()
