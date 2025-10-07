Biblioteca a descargar: pip install ultralytics

El dataset descargado de Kaggle solo tiene las imágenes. Se puede descargar en:

DATASET: https://drive.google.com/file/d/1-dIdLGGKH-F99NEMCQI6uI7FInM9XwVU/view?usp=sharing

Para usar YOLO 8, es necesario que el dataset tenga imágenes y anotaciones. El archivo generar_labels.py se creó para convertir el dataset de Kaggle en un dataset de Yolo. Se puede descargar en:

DATASET_YOLO: https://drive.google.com/file/d/1UaNPLI1bi7NiCSjJA5O9GOckt2qhglGW/view?usp=sharing

Para entrenar el modelo hay que poner el dataset_yolo en la misma ruta que el archivo entrenar_yolo.py, y modificar el archivo data.yaml para que apunte a esa ruta (no encontré cómo modificarlo para que use rutas relativas)

Si ejecutan el archivo entrenar_yolo.py y eso va a generar una carpeta con varios archivos. De esa carpeta me quedo con los archivos last.pt (el último modelo generado) y best.pt (el mejor modelo generado)

Si ejecutan el archivo predecir.py muestra el resultado de la predicción de algunos alimentos. 
