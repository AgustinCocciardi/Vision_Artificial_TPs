Biblioteca a descargar: pip install ultralytics

DATASET_YOLO: [https://drive.google.com/file/d/1UaNPLI1bi7NiCSjJA5O9GOckt2qhglGW/view?usp=sharing](https://drive.google.com/file/d/1fZqk7qdFZXrCkBSA3NMIsv0fyoS3UsbC/view?usp=drive_link)

Para entrenar el modelo hay que poner el dataset_yolo en la misma ruta que el archivo entrenar_yolo.py, y modificar el archivo data.yaml para que apunte a esa ruta (no encontré cómo modificarlo para que use rutas relativas)

Si ejecutan el archivo entrenar_yolo.py y eso va a generar una carpeta con varios archivos. De esa carpeta me quedo con los archivos last.pt (el último modelo generado) y best.pt (el mejor modelo generado)


