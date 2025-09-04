import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import mediapipe as mp
import time

# ====================
# RUTAS RELATIVAS
# ====================

base_dir = os.path.dirname(os.path.abspath(__file__))  # ruta base del script

dataset_dir = os.path.join(base_dir, "dataset", "train")  # carpeta con datos de entrenamiento
test_dir = os.path.join(base_dir, "dataset", "test")      # carpeta con datos de prueba
model_path = os.path.join(base_dir, "model_auto_moto.h5") # ruta donde se guarda/carga el modelo

# ====================
# ENTRENAMIENTO O CARGA DEL MODELO
# ====================

if os.path.exists(model_path):
    # Si ya existe un modelo entrenado, lo cargamos
    print("Cargando modelo existente...")
    model = tf.keras.models.load_model(model_path)
else:
    # Si no existe, se entrena desde cero
    print("Entrenando modelo...")
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)  # normalización y validación

    # Datos de entrenamiento
    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),   # redimensiona las imágenes
        batch_size=16,            # cantidad de imágenes por batch
        class_mode='binary',      # dos clases (autos, motos)
        subset='training',
        classes=['autos','motos'] # orden explícito de las clases
    )

    # Datos de validación
    val_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary',
        subset='validation',
        classes=['autos','motos']
    )

    # Definición de la CNN
    model = tf.keras.Sequential([
        # Capa convolucional: detecta bordes/patrones
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),  # reduce resolución manteniendo info relevante

        # Segunda capa convolucional más profunda
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        # Aplanamiento y capas densas para clasificación
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # salida binaria (Auto/Moto)
    ])

    # Compilación del modelo
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Entrenamiento
    model.fit(train_data, validation_data=val_data, epochs=5)

    # Guardar el modelo entrenado
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

# ====================
# CLASIFICAR PRUEBAS
# ====================

# Lista de imágenes en la carpeta test
images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith((".jpg", ".png"))]
index = 0  # índice de la imagen actual

# Inicialización de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)  # captura de cámara en vivo

# Tiempo que hay que mantener un gesto para cambiar imagen
GESTURE_HOLD_TIME = 5
gesture_start_time = None
gesture_type = None

# Función de predicción de una imagen
def predict_image(img_path):
    img = load_img(img_path, target_size=(128,128))  # carga y redimensiona
    img_array = img_to_array(img) / 255.0            # normaliza
    img_array = np.expand_dims(img_array, axis=0)    # añade dimensión batch
    pred = model.predict(img_array)[0][0]            # predicción (0=Auto, 1=Moto)
    label = "Moto" if pred > 0.5 else "Auto"
    prob_auto = 1 - pred
    prob_moto = pred
    return label, prob_auto, prob_moto

# Bucle principal
while True:
    # Cargar imagen actual y predecir
    img_path = images[index]
    label, prob_auto, prob_moto = predict_image(img_path)
    img_display = cv2.imread(img_path)

    # Mostrar predicciones sobre la imagen
    cv2.putText(img_display, f"Prediccion: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img_display, f"Prob Auto: {prob_auto*100:.1f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(img_display, f"Prob Moto: {prob_moto*100:.1f}%", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Clasificador Auto/Moto", img_display)

    # Leer frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar frame con MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_gesture = None

    # Si se detectan manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja landmarks en el frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas del pulgar e índice
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Gesto simple: pulgar arriba/abajo
            if thumb_tip.y < index_tip.y:
                detected_gesture = "up"
            elif thumb_tip.y > index_tip.y:
                detected_gesture = "down"

    # Control de tiempo: mantener gesto varios segundos
    if detected_gesture == gesture_type:
        if gesture_start_time is None:
            gesture_start_time = time.time()
        elif time.time() - gesture_start_time >= GESTURE_HOLD_TIME:
            if detected_gesture == "up":
                index = min(index + 1, len(images) - 1)
                print("👍 Siguiente imagen")
            elif detected_gesture == "down":
                index = max(index - 1, 0)
                print("👎 Imagen anterior")
            gesture_start_time = None
    else:
        gesture_type = detected_gesture
        gesture_start_time = time.time() if detected_gesture else None

    # Mostrar cámara con tracking de manos
    cv2.imshow("Camara", frame)

    # Tecla ESC para salir
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
