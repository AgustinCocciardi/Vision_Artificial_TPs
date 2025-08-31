import cv2
import numpy as np

# Callback vacío para las trackbars
def nothing(x): 
    pass

# Crear ventana de configuración con sliders (trackbars)
cv2.namedWindow("Ajustes")

# Umbral para la binarización
cv2.createTrackbar("Umbral", "Ajustes", 127, 255, nothing)
# Tamaño del kernel para operaciones morfológicas
cv2.createTrackbar("Kernel", "Ajustes", 1, 20, nothing)
# Umbral para aceptar coincidencia de estrella
cv2.createTrackbar("Match Thresh", "Ajustes", 20, 100, nothing)

# ------------------------------------------------------------------
# Función que crea imágenes de referencia para cada forma
# Devuelve el contorno de la figura
# ------------------------------------------------------------------
def create_ref_shape(shape="triangle"):
    img = np.zeros((300, 300), dtype=np.uint8)
    
    if shape == "triangle":
        pts = np.array([[150, 50], [50, 250], [250, 250]], np.int32)
        cv2.drawContours(img, [pts], 0, 255, -1)  
    
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (250, 250), 255, -1)
    
    elif shape == "star":
        pts = np.array([[150,20],[180,100],[260,100],[200,160],
                        [220,240],[150,190],[80,240],[100,160],
                        [40,100],[120,100]], np.int32)
        cv2.drawContours(img, [pts], 0, 255, -1)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

# ------------------------------------------------------------------
# Diccionario con contornos de referencia
# ------------------------------------------------------------------
ref_shapes = {
    "Estrella": create_ref_shape("star")
}

# ------------------------------------------------------------------
# Iniciar la webcam
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Valores desde sliders
    thresh_val = cv2.getTrackbarPos("Umbral", "Ajustes")
    kernel_size = cv2.getTrackbarPos("Kernel", "Ajustes")
    match_thresh = cv2.getTrackbarPos("Match Thresh", "Ajustes") / 100.0

    # Binarización
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Morfología para eliminar ruido
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contornos de la imagen procesada
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Aproximación del contorno (para contar vértices)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Default → Desconocido
        shape_name = "Desconocido"
        color = (0, 0, 255)  

        # Reglas de clasificación
        if len(approx) == 3:
            shape_name = "Triángulo"
            color = (0, 255, 0)

        elif len(approx) == 4:
            # Checar si es cuadrado o rectángulo
            x2, y2, w2, h2 = cv2.boundingRect(approx)
            aspect_ratio = float(w2) / h2
            if 0.9 <= aspect_ratio <= 1.1:
                shape_name = "Cuadrado"
                color = (0, 255, 0)
            else:
                shape_name = "Rectángulo"
                color = (0, 255, 0)

        elif len(approx) >= 8:
            # Usamos matchShapes solo para estrella
            score_star = cv2.matchShapes(cnt, ref_shapes["Estrella"], 1, 0.0)
            if score_star < match_thresh:
                shape_name = "Estrella"
                color = (0, 255, 0)

        # Dibujar contorno
        cv2.drawContours(frame, [cnt], -1, color, 2)

        # Mostrar etiqueta SIEMPRE (incluyendo "Desconocido")
        cv2.putText(frame, shape_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar resultados
    cv2.imshow("Resultado", frame)
    cv2.imshow("Thresh", morph)

    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
