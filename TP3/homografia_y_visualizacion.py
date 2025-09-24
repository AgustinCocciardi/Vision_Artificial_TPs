import cv2
import numpy as np
 
# ======================
# 1. Cargar calibración
# ======================
data = np.load("calibration_data.npz")
mtx = data["mtx"]
dist = data["dist"]
 
FRONTAL_SIZE = (300, 300)   # tamaño de la vista rectificada
GRID_SIZE = 3               # grilla de NxN
 
# ======================
# 2. Funciones auxiliares
# ======================
def draw_grid(img, H, grid_size=GRID_SIZE, color=(0,255,0)):
    """Dibuja una grilla NxN en la imagen original usando homografía inversa."""
    h, w = FRONTAL_SIZE
    step_x, step_y = w // grid_size, h // grid_size
 
    # Invertir H para llevar del plano frontal → imagen original
    H_inv = np.linalg.inv(H)
 
    for i in range(1, grid_size):
        # Líneas verticales
        pts = np.array([[i*step_x, 0], [i*step_x, h]], dtype=np.float32).reshape(-1,1,2)
        pts = cv2.perspectiveTransform(pts, H_inv)
        cv2.line(img, tuple(pts[0,0].astype(int)), tuple(pts[1,0].astype(int)), color, 2)
 
        # Líneas horizontales
        pts = np.array([[0, i*step_y], [w, i*step_y]], dtype=np.float32).reshape(-1,1,2)
        pts = cv2.perspectiveTransform(pts, H_inv)
        cv2.line(img, tuple(pts[0,0].astype(int)), tuple(pts[1,0].astype(int)), color, 2)
 
def compute_homography_from_clicks(frame):
    """Permite seleccionar 4 puntos con clics."""
    points_src = []
 
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_src) < 4:
            points_src.append((x,y))
            print(f"Punto {len(points_src)}: {x},{y}")
 
    clone = frame.copy()
    cv2.namedWindow("Selecciona 4 puntos")
    cv2.setMouseCallback("Selecciona 4 puntos", mouse_callback)
 
    while True:
        vis = clone.copy()
        for p in points_src:
            cv2.circle(vis, p, 5, (0,0,255), -1)
        cv2.imshow("Selecciona 4 puntos", vis)
        key = cv2.waitKey(1) & 0xFF
        if len(points_src) == 4:
            break
        if key != 255 and key != ord('h'):  # cualquier tecla aborta
            points_src = []
            break
 
    cv2.destroyWindow("Selecciona 4 puntos")
 
    if len(points_src) == 4:
        pts_src = np.array(points_src, dtype=np.float32)
        w, h = FRONTAL_SIZE
        pts_dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
        print("Homografía manual:\n", H)
        return H
    return None
 
def compute_homography_from_qr(frame):
    """Detecta QR y devuelve homografía si existe."""
    qr = cv2.QRCodeDetector()
    data, points, _ = qr.detectAndDecode(frame)
    if points is not None:
        pts_src = points[0].astype(np.float32)
        w, h = FRONTAL_SIZE
        pts_dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        H, _ = cv2.findHomography(pts_src, pts_dst)
        print("Homografía QR:\n", H)
        return H
    return None
 
# ======================
# 3. Cámara en vivo
# ======================
cap = cv2.VideoCapture(0)
H = None   # homografía actual
mode = "visualizacion"
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
 
    vis = undistorted.copy()
 
    if H is not None:
        # Dibuja grilla sobre la imagen
        draw_grid(vis, H)
 
        # Vista rectificada
        warped = cv2.warpPerspective(undistorted, H, FRONTAL_SIZE)
        cv2.imshow("Rectificado", warped)
 
    cv2.imshow("Visualizacion", vis)
 
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('h'):  # modo manual
        H_new = compute_homography_from_clicks(undistorted)
        if H_new is not None:
            H = H_new
    elif key == ord('q'):  # modo QR
        H_new = compute_homography_from_qr(undistorted)
        if H_new is not None:
            H = H_new
 
cap.release()
cv2.destroyAllWindows()