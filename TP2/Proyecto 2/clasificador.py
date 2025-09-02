import cv2
import numpy as np
import joblib

# Cargar modelo entrenado
modelo = joblib.load("modelo_figuras.pkl")

# Diccionario de etiquetas
etiquetas = {1: "Cuadrado", 2: "Triángulo", 3: "Estrella"}

# Captura desde la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a gris y binarizar
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        # Tomar el contorno más grande
        c = max(contornos, key=cv2.contourArea)

        if cv2.contourArea(c) > 500:  # evitar ruido muy pequeño
            # Calcular momentos
            momentos = cv2.moments(c)
            hu = cv2.HuMoments(momentos).flatten()

            # Transformación logarítmica para estabilidad
            hu = -np.sign(hu) * np.log10(np.abs(hu))

            # Predicción con el modelo
            pred = modelo.predict([hu])[0]

            # Dibujar el contorno
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

            # Mostrar nombre de la figura
            texto = etiquetas.get(pred, "Desconocido")
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(frame, texto, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar la ventana
    cv2.imshow("Detector de Figuras", frame)

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
