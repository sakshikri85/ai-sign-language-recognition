import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf

# ========= CONFIG =========
DATASET_PATH = "dataset"
MODEL_PATH = "gesture_model.h5"
IMG_SIZE = 64

# ========= Load class names =========
class_names = sorted(os.listdir(DATASET_PATH))
print("Class labels loaded:", class_names)

# ========= Load trained model =========
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)
print("Model loaded successfully!")

# ========= MediaPipe =========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ========= Start webcam =========
cap = cv2.VideoCapture(0)

print("\nPlace your hand in front of webcam")
print("Press Q to quit\n")

last_pred = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            xs = [int(lm.x * w) for lm in hand.landmark]
            ys = [int(lm.y * h) for lm in hand.landmark]
            x1, x2 = max(min(xs)-20, 0), min(max(xs)+20, w)
            y1, y2 = max(min(ys)-20, 0), min(max(ys)+20, h)

            roi = frame[y1:y2, x1:x2]

            try:
                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi = roi.astype("float32") / 255.0
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)
                idx = np.argmax(preds[0])
                last_pred = class_names[idx]

            except:
                pass

    if last_pred:
        cv2.putText(frame, f"Prediction: {last_pred}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Capture dataset ", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
