import cv2
import numpy as np
import os

def nothing(x):
    pass

# Ventana de ajustes
cv2.namedWindow("Ajustes")
cv2.createTrackbar("Umbral", "Ajustes", 127, 255, nothing)
cv2.createTrackbar("Kernel", "Ajustes", 1, 20, nothing)
cv2.createTrackbar("Match Thresh", "Ajustes", 20, 100, nothing)
cv2.createTrackbar("Area Min", "Ajustes", 500, 5000, nothing)

# Ruta de la carpeta actual (donde está este script en Proyecto 2)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta hacia Proyecto 1/Figuras (hermano de Proyecto 2)
ref_dir = os.path.join(base_dir, "..", "Proyecto 1", "Figuras")

print("Ruta absoluta de Figuras:", os.path.abspath(ref_dir))
print("¿Existe carpeta?", os.path.exists(ref_dir))

def load_reference_shape(filename):
    path = os.path.join(ref_dir, filename)
    print("Buscando:", os.path.abspath(path))  # debug
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

ref_shapes = {
    "Cuadrado": load_reference_shape("cuadrado.jpg"),
    "Triangulo": load_reference_shape("triangulo.jpg"),
    "Estrella": load_reference_shape("estrella.jpg"),
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
        if 0.9 <= aspect_ratio <= 1.1:
            score = cv2.matchShapes(cnt, refs["Cuadrado"], cv2.CONTOURS_MATCH_I1, 0.0)
            candidates["Cuadrado"] = score

    # Estrella
    elif n_vertices >= 8:
        if solidity < 0.9:
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

# Guardar el último contorno válido
last_detected_shape = None
last_detected_contour = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_val = cv2.getTrackbarPos("Umbral", "Ajustes")
    kernel_size = cv2.getTrackbarPos("Kernel", "Ajustes")
    match_thresh = cv2.getTrackbarPos("Match Thresh", "Ajustes") / 100.0
    area_min = cv2.getTrackbarPos("Area Min", "Ajustes")

    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < area_min:
            continue

        shape_name = classify_shape(cnt, ref_shapes, match_thresh)
        color = (0, 255, 0) if shape_name != "Desconocido" else (0, 0, 255)

        cv2.drawContours(frame, [cnt], -1, color, 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(frame, shape_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Guardar la última figura detectada válida
        if shape_name in ["Triangulo", "Cuadrado", "Estrella"]:
            last_detected_shape = shape_name
            last_detected_contour = cnt

    cv2.imshow("Resultado", frame)
    cv2.imshow("Thresh", morph)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Barra espaciadora
        if last_detected_shape is not None and last_detected_contour is not None:
            moments = cv2.moments(last_detected_contour)
            hu = cv2.HuMoments(moments).flatten()
            print(f"\nInvariantes de Hu para {last_detected_shape}:")
            for i, val in enumerate(hu, 1):
                print(f"Hu[{i}] = {val:.6e}")

cap.release()
cv2.destroyAllWindows()
