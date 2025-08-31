import cv2
import numpy as np
import os

def nothing(x):
    pass

# Ventana de ajustes
cv2.namedWindow("Ajustes")
cv2.createTrackbar("Umbral", "Ajustes", 127, 255, nothing)
cv2.createTrackbar("Kernel", "Ajustes", 1, 20, nothing)
cv2.createTrackbar("Match Thresh", "Ajustes", 20, 100, nothing)  # umbral global para todas las figuras

# ------------------------------------------------------------------
# Cargar imágenes de referencia
# ------------------------------------------------------------------
def load_reference_shape(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "Figuras")
    path = os.path.join(folder, filename)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No se detectaron contornos en la imagen: {path}")
    return contours[0]

ref_shapes = {
    "Triangulo": load_reference_shape("triangulo.JPG"),
    "Cuadrado": load_reference_shape("cuadrado.JPG"),
    "Estrella": load_reference_shape("estrella.JPG")
}

# ------------------------------------------------------------------
# Clasificación con vértices + matchShapes + convexidad
# ------------------------------------------------------------------
def classify_shape(cnt, refs, match_thresh):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    n_vertices = len(approx)
    area = cv2.contourArea(cnt)
    solidity = float(area) / cv2.contourArea(cv2.convexHull(cnt))

    candidates = {}

    # Triángulo
    if n_vertices == 3:
        score = cv2.matchShapes(cnt, refs["Triangulo"], cv2.CONTOURS_MATCH_I1, 0.0)
        candidates["Triangulo"] = score

    # Cuadrado
    elif n_vertices == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if 0.9 <= aspect_ratio <= 1.1:  # solo cuadrados
            score = cv2.matchShapes(cnt, refs["Cuadrado"], cv2.CONTOURS_MATCH_I1, 0.0)
            candidates["Cuadrado"] = score

    # Estrella
    elif n_vertices >= 8:
        if solidity < 0.9:  # evitar círculos/figuras convexas
            score = cv2.matchShapes(cnt, refs["Estrella"], cv2.CONTOURS_MATCH_I1, 0.0)
            candidates["Estrella"] = score

    if not candidates:
        return "Desconocido"

    best_shape = min(candidates, key=candidates.get)
    if candidates[best_shape] > match_thresh:
        return "Desconocido"
    return best_shape

# ------------------------------------------------------------------
# Captura de webcam
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_val = cv2.getTrackbarPos("Umbral", "Ajustes")
    kernel_size = cv2.getTrackbarPos("Kernel", "Ajustes")
    match_thresh = cv2.getTrackbarPos("Match Thresh", "Ajustes") / 100.0

    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue

        shape_name = classify_shape(cnt, ref_shapes, match_thresh)
        color = (0, 255, 0) if shape_name != "Desconocido" else (0, 0, 255)

        cv2.drawContours(frame, [cnt], -1, color, 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(frame, shape_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Resultado", frame)
    cv2.imshow("Thresh", morph)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
