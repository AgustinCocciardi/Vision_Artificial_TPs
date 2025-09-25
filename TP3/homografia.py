"""
homografia_webcam.py

Uso:
    - Ejecutar: python homografia_webcam.py
    - Teclas:
        q : entrar en modo de detección automática por QR
        h : entrar en modo de homografía asistida (clics del mouse)
        ESC o 'x' : salir
    - En modo QR: si se detecta un QR, al presionar cualquier tecla se calcula la homografía y vuelve a visualización
    - En modo asistido: hace 4 clics en la ventana principal sobre los 4 vértices del cuadrado (orden libre).
                     Tras el 4º clic se calcula la homografía y vuelve a visualización.
    - Visualización:
        - Se dibuja una grilla (por defecto 3x3) proyectada por la homografía
        - Se abre otra ventana con la vista frontal (warp) del cuadrado
"""

import cv2
import numpy as np
import os

# --- Configuración ---
CAMERA_ID = 0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(SCRIPT_DIR, "calibration_data.npz")  # tu archivo (mtx, dist, ...)
GRID_CELLS = 3   # dibujar grilla NxN (por ejemplo 3x3)
FRONTAL_SIZE = (600, 600)  # tamaño de la ventana frontal (pixeles)
WINDOW_NAME = "Homografia - Webcam"
FRONT_WINDOW = "Vista frontal (warp)"
INFO_WINDOW_POS = (10, 10)

# estado global
mode = "view"  # view | qr | assisted
homography = None
src_square = np.array([[0, 0],
                       [FRONTAL_SIZE[0]-1, 0],
                       [FRONTAL_SIZE[0]-1, FRONTAL_SIZE[1]-1],
                       [0, FRONTAL_SIZE[1]-1]], dtype=np.float32)  # vista frontal: esquina TL, TR, BR, BL
assisted_points = []

# Intentar cargar calibración
cam_mtx = None
cam_dist = None
if os.path.exists(CALIB_FILE):
    try:
        npz = np.load(CALIB_FILE, allow_pickle=True)
        cam_mtx = npz.get('mtx', None)
        cam_dist = npz.get('dist', None)
        print(f"Cargando calibración desde {CALIB_FILE}")
    except Exception as e:
        print("No se pudo cargar calibración:", e)
else:
    print(f"No se encontró {CALIB_FILE}. Se trabajará sin corrección de distorsión.")

# mouse callback
def mouse_callback(event, x, y, flags, param):
    global assisted_points, mode, homography
    if mode != "assisted":
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(assisted_points) < 4:
            assisted_points.append((x, y))
            print(f"Clic {len(assisted_points)}: {(x, y)}")
        if len(assisted_points) == 4:
            # Al alcanzar 4 puntos, calculamos homografía desde la vista frontal (src_square) -> puntos seleccionados
            dst = np.array(assisted_points, dtype=np.float32)
            H, status = cv2.findHomography(src_square, dst, method=0)
            if H is not None:
                homography = H
                print("Homografía calculada desde los 4 clics.")
            else:
                print("No se pudo calcular la homografía con los 4 clics.")
            mode = "view"
            assisted_points = []

# proyección de grilla frontal y dibujo sobre imagen
def draw_projected_grid(img, H, cells=3, color=(0,255,0), thickness=2):
    if H is None:
        return img
    h, w = FRONTAL_SIZE[1], FRONTAL_SIZE[0]
    # generar líneas de grilla en la vista frontal
    # coordenadas en el plano frontal (x, y)
    pts = []
    for i in range(cells+1):
        # horizontales: y constante
        y = i * (h / cells)
        line = np.array([[0, y], [w-1, y]], dtype=np.float32)
        pts.append(line)
    for j in range(cells+1):
        # verticales: x constante
        x = j * (w / cells)
        line = np.array([[x, 0], [x, h-1]], dtype=np.float32)
        pts.append(line)
    # proyectar cada línea con H^-1? No: tenemos H que mapea frontal->imagen (si usaste src->dst). 
    # En este script H mapea frontal->imagen (si fue calculada así). Proyectamos usando perspectiveTransform.
    for line in pts:
        src = line.reshape(-1,1,2)
        dst = cv2.perspectiveTransform(src, H)
        p1 = tuple(dst[0,0].astype(int))
        p2 = tuple(dst[1,0].astype(int))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    return img

# dibujar los puntos asistidos
def draw_assisted_points(img, points):
    for i, p in enumerate(points):
        cv2.circle(img, p, 6, (0,0,255), -1)
        cv2.putText(img, str(i+1), (p[0]+8, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# detectar QR y devolver los 4 vértices en el orden adecuado (si no detecta -> None)
def detect_qr_corners(gray):
    detector = cv2.QRCodeDetector()
    # detect devuelve retval, points
    status, points = detector.detect(gray)
    if status and points is not None:
        # points es (4,1,2) o (4,2). Normalizar
        pts = np.array(points).reshape(-1,2).astype(np.float32)
        # Asegurar que hay 4 puntos
        if pts.shape[0] == 4:
            return pts
    return None

# función para intentar undistort
def undistort_frame(frame, mtx, dist):
    if mtx is None or dist is None: return frame
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    und = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    return und

# inicializar captura
cap = cv2.VideoCapture(CAMERA_ID)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("Iniciando captura. Presiona 'q' para modo QR, 'h' para modo asistido, ESC o 'x' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer de la cámara.")
        break

    # corregir distorsión si hay calibracion
    frame_proc = undistort_frame(frame, cam_mtx, cam_dist)

    display = frame_proc.copy()
    key = cv2.waitKey(1) & 0xFF

    # Mostrar info de modo en la imagen
    cv2.putText(display, f"Modo: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(display, "q=QR  h=asistida  x/ESC=salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    if mode == "view":
        # dibujar grilla proyectada si hay homografía
        if homography is not None:
            try:
                display = draw_projected_grid(display, homography, cells=GRID_CELLS)
                # también dibujar el contorno proyectado del cuadrado
                dst = cv2.perspectiveTransform(src_square.reshape(-1,1,2), homography).reshape(-1,2).astype(int)
                for i in range(4):
                    p1 = tuple(dst[i])
                    p2 = tuple(dst[(i+1)%4])
                    cv2.line(display, p1, p2, (255,0,0), 2)
            except Exception as e:
                print("Error al dibujar proyección:", e)

        # si hay homografia, mostrar warp (vista frontal)
        if homography is not None:
            try:
                # La homografía que tenemos mapea src_square (frontal) -> imagen. Para warpPerspective necesitamos H_inv que mapea imagen->frontal.
                H_inv = np.linalg.inv(homography)
                warped = cv2.warpPerspective(frame_proc, H_inv, FRONTAL_SIZE)
                cv2.imshow(FRONT_WINDOW, warped)
            except Exception as e:
                print("Error al generar warp:", e)
        else:
            # si no hay homografía cerrar ventana frontal si existe
            if cv2.getWindowProperty(FRONT_WINDOW, cv2.WND_PROP_VISIBLE) >= 0:
                try:
                    cv2.destroyWindow(FRONT_WINDOW)
                except:
                    pass

    elif mode == "qr":
        # detectar QR en la imagen (usar gris)
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        pts = detect_qr_corners(gray)
        if pts is not None:
            pts_int = pts.astype(int)
            # dibujar polígono
            for i in range(4):
                p1 = tuple(pts_int[i])
                p2 = tuple(pts_int[(i+1)%4])
                cv2.line(display, p1, p2, (0,255,255), 2)
            cv2.putText(display, "QR detectado. Presione cualquier tecla para computar homografía", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            # si el usuario presiona cualquier tecla (excepto nada), computar homografía
            if key != 255:
                # Queremos H que mapee src_square (frontal) -> pts (imagen)
                dst = pts.astype(np.float32)
                H, status = cv2.findHomography(src_square, dst, method=0)
                if H is not None:
                    homography = H
                    print("Homografía calculada desde QR.")
                else:
                    print("No se pudo calcular la homografía desde el QR.")
                mode = "view"
                # limpiar key para evitar efecto rebote
                key = 255
        else:
            cv2.putText(display, "No se detecta QR. Presione cualquier tecla para volver.", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,150), 2)
            if key != 255:
                mode = "view"
                key = 255

    elif mode == "assisted":
        cv2.putText(display, "Modo asistido: haga 4 clics en los vértices del cuadrado (izq-izq?)", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 2)
        draw_assisted_points(display, assisted_points)
        # posibilidad de abortar con cualquier tecla: según la especificación
        if key != 255:
            # abortar y volver a view manteniendo homografía anterior
            assisted_points = []
            mode = "view"

    # procesamiento de teclas global (después del modo)
    if key == ord('q'):
        mode = "qr"
        assisted_points = []
        print("Entrando en modo QR.")
    elif key == ord('h'):
        mode = "assisted"
        assisted_points = []
        print("Entrando en modo asistido (4 clics).")
    elif key == 27 or key == ord('x'):  # ESC o 'x'
        print("Saliendo.")
        break

    cv2.imshow(WINDOW_NAME, display)

# limpieza
cap.release()
cv2.destroyAllWindows()
