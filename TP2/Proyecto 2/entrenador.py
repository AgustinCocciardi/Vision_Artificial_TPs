import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Cargar dataset (ajustá la ruta a tu archivo CSV o Excel)
# Ejemplo: dataset.csv con columnas hu1...hu7 y etiqueta
df = pd.read_csv("dataset.csv")

# Definir variables independientes (X) y dependiente (y)
X = df[['hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7']]
y = df['etiqueta']

# Crear y entrenar el modelo
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X, y)

# Guardar el modelo en un archivo
joblib.dump(modelo, "modelo_figuras.pkl")

print("✅ Modelo entrenado y guardado en 'modelo_figuras.pkl'")
