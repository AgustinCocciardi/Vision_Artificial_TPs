import cv2
import numpy as np
import glob

# Tamaño del patrón de ajedrez (número de esquinas internas por fila y columna)
chessboard_size = (7, 7)  # 9 esquinas horizontales, 6 verticales
square_size = 25  # tamaño del cuadrado en milímetros (puede ser cualquier unidad)

# Preparar puntos 3D en el espacio real (0,0,0), (1,0,0), (2,0,0)...
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays para guardar puntos 3D y 2D de todas las imágenes
objpoints = []  # puntos 3D en el mundo real
imgpoints = []  # puntos 2D en la imagen

# Captura imágenes de la cámara para calibrar
cap = cv2.VideoCapture(0)

print("Presiona 's' para guardar imagen con el patrón detectado, 'q' para salir.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Encontrar esquinas del ajedrez
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret_corners:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_corners)

    cv2.imshow('Calibración de cámara', frame)
    key = cv2.waitKey(1)

    if key == ord('s') and ret_corners:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        print("Imagen guardada para calibración.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibrar cámara
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\nMatriz de la cámara (intrínsecos):\n", mtx)
    print("\nCoeficientes de distorsión:\n", dist)

    # Guardar parámetros
    np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("\nCalibración guardada en calibration_data.npz")
else:
    print("No se capturaron suficientes imágenes para calibrar.")
