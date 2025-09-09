import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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

    # Definir pipeline: escalado + modelo
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    # Definir hiperparámetros para grid search
    param_grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5]
    }

    # Configurar GridSearchCV con validación cruzada
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        scoring="accuracy",  # métrica de evaluación
        n_jobs=-1            # usar todos los núcleos disponibles
    )

    # Entrenar con búsqueda de hiperparámetros
    grid_search.fit(X, y)

    # Mejor modelo
    best_model = grid_search.best_estimator_

    # Guardar el mejor modelo en la misma ruta del script
    ruta_guardado = os.path.join(script_dir, MODEL_FILE)
    joblib.dump(best_model, ruta_guardado)

    print(f"✅ Modelo entrenado y guardado en: {ruta_guardado}")
    print("Mejores hiperparámetros encontrados:", grid_search.best_params_)
    print("Mejor accuracy en validación cruzada:", grid_search.best_score_)

if __name__ == "__main__":
    main()
