"""
Driver Safety Monitor - Aplicación Principal
"""

# --- IMPORTS ---
import cv2 as cv
import numpy as np
import mediapipe as mp
from time import time
from collections import deque
from dataclasses import dataclass
import math
import os
from ultralytics import YOLO

# --- UI ---
from ui_manager_v3 import (
    UIManager, 
    FusionWeights, 
    fuse_scores, 
    show_splash_screen,
    CAM_WIDTH,
    CAM_HEIGHT
)

# =======================================================
# 0. CARGAR MODELOS YOLO (celular y cinturón)
# =======================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CELULAR_MODEL_PATH = os.path.join(SCRIPT_DIR, "celular.pt")
CINTURON_MODEL_PATH = os.path.join(SCRIPT_DIR, "cinturon.pt")

yolo_phone = YOLO(CELULAR_MODEL_PATH)
yolo_belt  = YOLO(CINTURON_MODEL_PATH)


# =======================================================
# 1. MEDIA PIPE HELPERS
# =======================================================

class MPHelpers:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(max_num_hands=2)
        self.pose = mp.solutions.pose.Pose(model_complexity=1)

    def process(self, bgr):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        face = self.face_mesh.process(rgb)
        hands = self.hands.process(rgb)
        pose = self.pose.process(rgb)
        return face, hands, pose


# =======================================================
# 2. SOMNOLENCIA: EAR / MAR / PERCLOS / BLINK RATE
# =======================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

def euclidean(a, b):
    return np.linalg.norm(a - b)

def compute_EAR(landmarks, eye_ids):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])
    A = euclidean(pts[eye_ids[1]], pts[eye_ids[5]])
    B = euclidean(pts[eye_ids[2]], pts[eye_ids[4]])
    C = euclidean(pts[eye_ids[0]], pts[eye_ids[3]])
    EAR = (A + B) / (2.0 * C)
    return EAR

def compute_MAR(landmarks, mouth_ids):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])
    A = euclidean(pts[mouth_ids[0]], pts[mouth_ids[1]])
    C = euclidean(pts[mouth_ids[2]], pts[mouth_ids[3]])
    return A / C


class PERCLOS:
    def __init__(self, window_seconds=60):
        self.window = window_seconds
        self.data = deque()

    def update(self, closed, t):
        self.data.append((t, closed))
        while self.data and self.data[0][0] < t - self.window:
            self.data.popleft()

    def value(self):
        if not self.data:
            return 0
        closed = sum(1 for _, c in self.data if c)
        return closed / len(self.data)


class BlinkRate:
    def __init__(self, window=15):
        self.window = window
        self.blinks = deque()
        self.prev_closed = False

    def update(self, closed, t):
        if closed and not self.prev_closed:
            self.blinks.append(t)
        self.prev_closed = closed
        while self.blinks and self.blinks[0] < t - self.window:
            self.blinks.popleft()

    def value(self):
        return len(self.blinks) / self.window


# =======================================================
# 3. CINTURÓN — YOLO
# =======================================================

def detect_belt_yolo(frame, min_conf=0.5):
    """
    Devuelve:
    0.0  si encuentra cinturón
    100 si no encuentra
    """
    res = yolo_belt(frame, verbose=False)[0]

    for b in res.boxes:
        if float(b.conf[0]) >= min_conf:
            return 0.0  # cinturón detectado

    return 100.0  # no detectado → riesgo


# =======================================================
# 4. CELULAR — YOLO
# =======================================================

def detect_phone_yolo(frame, min_conf=0.5):
    """
    Tu modelo tiene cls=1 → celular
    Devuelve:
    100 si detecta celular,
    0 si no.
    """
    res = yolo_phone(frame, verbose=False)[0]

    for b in res.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])

        if cls == 1 and conf >= min_conf:
            return 100.0

    return 0.0


# =======================================================
# 5. MAIN
# =======================================================

def main():

    show_splash_screen(3000)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    mp_h = MPHelpers()
    perclos = PERCLOS()
    blinks = BlinkRate()

    ui = UIManager(col1_x=16, col2_x=340)
    weights = FusionWeights(0.5, 0.25, 0.25)

    last_t = time()

    while True:
        ok, frame = cap.read()
        if not ok: 
            break

        frame = cv.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        now = time()
        fps = 1.0 / (now - last_t)
        last_t = now

        canvas = ui.create_canvas(frame)

        face, hands, pose = mp_h.process(frame)

        EAR, MAR = 0, 0
        eyes_closed = False
        yawn = False
        score_somn = 0

        # ----------------------------------------
        # SOMNOLENCIA REAL
        # ----------------------------------------
        if face.multi_face_landmarks:
            lm = face.multi_face_landmarks[0].landmark

            EAR_L = compute_EAR(lm, LEFT_EYE)
            EAR_R = compute_EAR(lm, RIGHT_EYE)
            EAR = (EAR_L + EAR_R) / 2

            MAR = compute_MAR(lm, MOUTH)

            eyes_closed = EAR < 0.21
            yawn = MAR > 0.7

            perclos.update(eyes_closed, now)
            blinks.update(eyes_closed, now)

            score_somn = (
                perclos.value() * 60 +
                (20 if yawn else 0) +
                (20 if blinks.value() > 0.35 else 0)
            )
            score_somn = min(score_somn, 100)

        # ----------------------------------------
        # CINTURÓN — YOLO
        # ----------------------------------------
        score_belt = detect_belt_yolo(frame)

        # ----------------------------------------
        # CELULAR — YOLO
        # ----------------------------------------
        score_phone = detect_phone_yolo(frame)

        # ----------------------------------------
        # FUSIÓN
        score_global = fuse_scores(score_somn, score_belt, score_phone, weights)

        if score_global > 50: state = 2
        elif score_global > 25: state = 1
        else: state = 0

        # ----------------------------------------
        # UI
        # ----------------------------------------
        ui.reset_positions()

        ui.draw_bar_col1(canvas, score_somn, "Somnolencia")
        ui.draw_bar_col1(canvas, score_belt, "Cinturon")
        ui.draw_bar_col1(canvas, score_phone, "Celular")
        ui.draw_bar_col1(canvas, score_global, "Riesgo Global")

        ui.put_kv_col2(canvas, f"FPS: {fps:.1f}")
        ui.put_kv_col2(canvas, f"EAR: {EAR:.3f} MAR: {MAR:.3f}")
        ui.put_kv_col2(canvas, f"PERCLOS: {perclos.value():.2f}")
        ui.put_kv_col2(canvas, f"Blink rate: {blinks.value():.2f}/s")

        ui.draw_alerts(canvas, state)

        cv.imshow(ui.window_name, canvas)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
