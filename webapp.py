import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf   # ✅ Python 3.11 SAFE import

# =====================================================
# CONFIG
# =====================================================
DATASET_DIR = "dataset"
MODEL_PATH = "gesture_model.h5"
LABELS_PATH = "labels.npy"

IMG_SIZE = 64          # MUST MATCH TRAINING
HOLD_THRESHOLD = 6

st.set_page_config(page_title="AI Sign Language", layout="wide")

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)

def list_classes():
    if not os.path.exists(DATASET_DIR):
        return []
    return sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

def count_images():
    stats = {}
    for cls in list_classes():
        p = os.path.join(DATASET_DIR, cls)
        stats[cls] = len([
            f for f in os.listdir(p)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
    return stats

# =====================================================
# HEADER & SIDEBAR
# =====================================================
ensure_dirs()
st.title("🖐️ AI Sign Language Recognition System")

mode = st.sidebar.radio(
    "Select Mode",
    ["🏠 Home", "📸 Capture Dataset", "🧠 Train Model", "🔮 Realtime Recognition"]
)

# =====================================================
# HOME
# =====================================================
if mode.startswith("🏠"):
    st.header("Dashboard Overview")

    classes = list_classes()
    st.write("### Available Gesture Classes:", classes)

    stats = count_images()
    if stats:
        fig, ax = plt.subplots()
        ax.bar(stats.keys(), stats.values())
        ax.set_ylabel("Number of Images")
        ax.set_title("Dataset Distribution")
        st.pyplot(fig)
    else:
        st.info("No dataset available. Capture images first.")

# =====================================================
# DATASET CAPTURE
# =====================================================
if mode.startswith("📸"):
    st.header("Capture Dataset Using Webcam")

    gesture_name = st.text_input("Gesture Name (Class Label)", "A")
    num_images = st.slider("Number of Images", 20, 500, 120)
    start_btn = st.button("Start Capture")

    preview = st.empty()

    if start_btn:
        save_dir = os.path.join(DATASET_DIR, gesture_name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0
        st.success("Capturing images...")

        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            x1, y1 = w//2 - 130, h//2 - 130
            x2, y2 = w//2 + 130, h//2 + 130
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            roi = frame[y1:y2, x1:x2]
            fname = os.path.join(save_dir, f"{gesture_name}_{count:04}.jpg")
            cv2.imwrite(fname, roi)

            preview.image(frame[:, :, ::-1])
            count += 1

        cap.release()
        st.success(f"{count} images captured successfully")

# =====================================================
# TRAIN MODEL INFO
# =====================================================
if mode.startswith("🧠"):
    st.header("Train CNN Model")

    st.code("python train_model.py")

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        labels = np.load(LABELS_PATH, allow_pickle=True)
        st.success("Model & Labels found")
        st.write("Classes:", list(labels))
    else:
        st.warning("Model not found. Train the model first.")

# =====================================================
# REALTIME RECOGNITION
# =====================================================
if mode.startswith("🔮"):
    st.header("Realtime Sign Language Recognition")

    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Train model first.")
        st.stop()

    # ✅ SAFE MODEL LOAD (Python 3.11)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

    if os.path.exists(LABELS_PATH):
        class_names = list(np.load(LABELS_PATH, allow_pickle=True))
    else:
        class_names = list_classes()

    st.success("Model loaded successfully")
    st.write("Recognized Classes:", class_names)

    start = st.button("Start Recognition")
    stop = st.button("Stop Recognition")

    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    frame_window = st.empty()
    text_box = st.empty()

    CONF_THRESH = 0.7

    if st.session_state.run:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1)
        draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        sentence = ""
        hold_count = 0
        last_pred = ""

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pred_text = "..."

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]

                x1, x2 = max(min(xs)-20, 0), min(max(xs)+20, w)
                y1, y2 = max(min(ys)-20, 0), min(max(ys)+20, h)

                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                    roi = roi.astype("float32") / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    preds = model.predict(roi, verbose=0)[0]
                    idx = np.argmax(preds)
                    prob = preds[idx]

                    if prob >= CONF_THRESH:
                        pred = class_names[idx]
                        pred_text = pred

                        if pred == last_pred:
                            hold_count += 1
                        else:
                            hold_count = 1
                            last_pred = pred

                        if hold_count >= HOLD_THRESHOLD:
                            sentence += pred
                            hold_count = 0
                    else:
                        last_pred = ""
                        hold_count = 0

            frame_window.image(frame[:, :, ::-1])
            text_box.markdown(
                f"### **Detected Gesture:** `{pred_text}`\n\n"
                f"### **Sentence:** `{sentence}`"
            )

        cap.release()
        hands.close()
        st.success("Recognition stopped")
