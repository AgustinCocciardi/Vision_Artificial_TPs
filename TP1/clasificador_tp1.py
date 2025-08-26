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

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join(base_dir, "dataset", "train")
test_dir = os.path.join(base_dir, "dataset", "test")  # carpeta actualizada
model_path = os.path.join(base_dir, "model_auto_moto.h5")

# ====================
# ENTRENAMIENTO O CARGA DEL MODELO
# ====================

if os.path.exists(model_path):
    print("Cargando modelo existente...")
    model = tf.keras.models.load_model(model_path)
else:
    print("Entrenando modelo...")
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary',
        subset='training',
        classes=['autos','motos']
    )

    val_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='binary',
        subset='validation',
        classes=['autos','motos']
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_data, validation_data=val_data, epochs=5)
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

# ====================
# CLASIFICAR PRUEBAS
# ====================

images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith((".jpg", ".png"))]
index = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

GESTURE_HOLD_TIME = 5  # segundos necesarios para pasar la imagen
gesture_start_time = None
gesture_type = None

def predict_image(img_path):
    img = load_img(img_path, target_size=(128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    label = "Moto" if pred > 0.5 else "Auto"
    prob_auto = 1 - pred
    prob_moto = pred
    return label, prob_auto, prob_moto

while True:
    img_path = images[index]
    label, prob_auto, prob_moto = predict_image(img_path)
    img_display = cv2.imread(img_path)

    cv2.putText(img_display, f"Prediccion: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img_display, f"Prob Auto: {prob_auto*100:.1f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(img_display, f"Prob Moto: {prob_moto*100:.1f}%", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Clasificador Auto/Moto", img_display)

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            if thumb_tip.y < index_tip.y:
                detected_gesture = "up"
            elif thumb_tip.y > index_tip.y:
                detected_gesture = "down"

    # Control de tiempo para mantener gesto
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

    cv2.imshow("Camara", frame)

    if cv2.waitKey(20) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
