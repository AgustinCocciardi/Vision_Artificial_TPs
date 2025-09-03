import os
import cv2
import numpy as np
import joblib

# ============================
# Configuración inicial
# ============================
def nothing(x):
    pass

# Ventana de ajustes
cv2.namedWindow("Ajustes")
cv2.createTrackbar("Umbral", "Ajustes", 127, 255, nothing)
cv2.createTrackbar("Kernel", "Ajustes", 1, 20, nothing)
cv2.createTrackbar("Area Min", "Ajustes", 500, 5000, nothing)

# Carpeta donde está tu script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta absoluta al modelo
modelo_path = os.path.join(script_dir, "modelo_figuras.pkl")

# Cargar modelo entrenado
modelo = joblib.load(modelo_path)

# Diccionario de etiquetas
etiquetas = {1: "Cuadrado", 2: "Triangulo", 3: "Estrella"}

# ============================
# Captura desde la webcam
# ============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Leer parámetros desde trackbars
    thresh_val = cv2.getTrackbarPos("Umbral", "Ajustes")
    kernel_size = cv2.getTrackbarPos("Kernel", "Ajustes")
    area_min = cv2.getTrackbarPos("Area Min", "Ajustes")

    # Binarización y operaciones morfológicas
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos
    contornos, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        if cv2.contourArea(cnt) < area_min:
            continue

        # Calcular invariantes de Hu
        momentos = cv2.moments(cnt)
        hu = cv2.HuMoments(momentos).flatten()

        # Transformación logarítmica
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        # Predicción con el modelo
        pred = modelo.predict([hu])[0]
        texto = etiquetas.get(pred, "Desconocido")

        # Dibujar contorno y etiqueta
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(frame, texto, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Mostrar resultados
    cv2.imshow("Resultado", frame)
    cv2.imshow("Thresh", morph)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
