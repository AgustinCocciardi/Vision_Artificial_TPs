import subprocess
import time
import os
import cv2
import face_recognition
import pickle
import numpy as np

# =============================
# CONFIGURACIÓN Y RUTAS
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas exactas indicadas
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "owner_embeddings.pkl")
DRIVER_MONITOR = os.path.join(BASE_DIR, "driver_safety_monitor.py")

# Nombre a mostrar si se reconoce (puedes cambiarlo)
OWNER_NAME = "Conductor Autorizado"

# ============================================
# FUNCIONES
# ============================================

def load_embeddings():
    """Solo carga el archivo .pkl existente."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return None
    
    print(f"[INFO] Cargando archivo de usuarios: {EMBEDDINGS_FILE}")
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)

def verify_user(known_embeddings_list):
    """
    Abre la cámara y compara con los embeddings cargados.
    Retorna True si reconoce, False si no.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se puede acceder a la cámara.")
        return False

    print("\n[INFO] Escaneando rostro...")
    
    start_time = time.time()
    recognized = False
    
    # Tiempo máximo de intento (ej. 10 segundos)
    # Si en 10 seg no reconoce, asume que no es la persona.
    MAX_DURATION = 10 

    while (time.time() - start_time) < MAX_DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar para procesar más rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Buscar caras
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        match_found = False
        
        for face_encoding in face_encodings:
            # Comparar con la lista cargada del .pkl
            # tolerance=0.5 es lo estándar, bájalo a 0.4 si es muy permisivo
            matches = face_recognition.compare_faces(known_embeddings_list, face_encoding, tolerance=0.5)
            
            if True in matches:
                match_found = True
                break
        
        # Feedback visual
        if match_found:
            cv2.putText(frame, "VERIFICADO", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Verificacion", frame)
            cv2.waitKey(500) # Pequeña pausa para ver el mensaje verde
            recognized = True
            break
        else:
            cv2.putText(frame, "Buscando...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Verificacion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return recognized

# =====================================================
# ORQUESTADOR PRINCIPAL
# =====================================================
def main():
    print("==========================================")
    print(" INICIANDO SISTEMA DE SEGURIDAD VEHICULAR")
    print("==========================================\n")

    # 1. Cargar el archivo .pkl
    embeddings = load_embeddings()

    if embeddings is None:
        print(f"[ERROR CRÍTICO] No se encontró el archivo: {EMBEDDINGS_FILE}")
        print("El sistema no puede verificar identidad, pero iniciará el monitor por seguridad.")
        # No intentamos enrolar, solo avisamos.
    else:
        # 2. Intentar reconocer
        is_recognized = verify_user(embeddings)

        if is_recognized:
            print("\n" + "="*40)
            print(f" [OK] PERSONA RECONOCIDA: {OWNER_NAME}")
            print("="*40 + "\n")
        else:
            print("\n" + "!"*40)
            print(" [ATENCIÓN] PERSONA NO RECONOCIDA")
            print("!"*40 + "\n")

    # 3. Pasar SIEMPRE al Driver Safety Monitor
    print(f"[INFO] Ejecutando monitor de seguridad ({DRIVER_MONITOR})...")
    time.sleep(2)
    
    if os.path.exists(DRIVER_MONITOR):
        subprocess.run(["python", DRIVER_MONITOR])
    else:
        print(f"[ERROR] No se encuentra el archivo {DRIVER_MONITOR}")

if __name__ == "__main__":
    main()