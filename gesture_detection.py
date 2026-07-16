import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import pyttsx3
import threading
import numpy as np

# ---  Voice Setup ---
_speaking = False

def speak(text):
    global _speaking
    if _speaking:
        return
    def run():
        global _speaking
        _speaking = True
        try:
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 150)
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()
        except:
            pass
        _speaking = False
    threading.Thread(target=run, daemon=True).start()

# ---  Custom Symbol Drawing ---
BRAILLE_DOTS = {
    "zero":    [],
    "one":     [(0,0)],
    "two":     [(0,0),(0,1)],
    "three":   [(0,0),(1,0),(0,1)],
    "four":    [(0,0),(1,0),(0,1),(1,1)],
    "five":    [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "thumbsup":   [(0,0),(0,2),(1,1)],
    "call me":    [(0,0),(1,0),(1,2)],
    "rock on":    [(0,0),(1,0),(0,1),(1,2)],
    "ok":         [(0,1),(1,1),(0,2),(1,2)],
    "unknown":    [(0,0),(1,1),(0,2),(1,0),(1,2)],
}

def draw_braille_symbol(frame, gesture_name, x, y):
    dots = BRAILLE_DOTS.get(gesture_name, BRAILLE_DOTS["unknown"])
    cv2.rectangle(frame, (x-5, y-5), (x+45, y+75), (30, 30, 30), -1)
    cv2.rectangle(frame, (x-5, y-5), (x+45, y+75), (100, 100, 100), 1)
    for col in range(2):
        for row in range(3):
            cx = x + col * 20 + 10
            cy = y + row * 22 + 10
            if (col, row) in dots:
                cv2.circle(frame, (cx, cy), 7, (0, 100, 255), -1)
                cv2.circle(frame, (cx, cy), 7, (0, 150, 255), 1)
            else:
                cv2.circle(frame, (cx, cy), 5, (60, 60, 60), -1)

# ---  MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# ---  Gesture Detection ---
def detect_gesture(lm_list, fingers, label):
    thumb, index, middle, ring, pinky = fingers
    total = sum(fingers)

    if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        if lm_list[4][1] < lm_list[3][1] < lm_list[2][1]:
            return "thumbsup"

    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4] and not fingers[0]:
        return "victory"

    if fingers[0] and fingers[4] and not fingers[1] and not fingers[2] and not fingers[3]:
        return "call me"

    if fingers[1] and fingers[4] and not fingers[2] and not fingers[3]:
        return "rock on"

    thumb_tip = lm_list[4]
    index_tip = lm_list[8]
    dist = np.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
    if dist < 40 and fingers[2] and fingers[3] and fingers[4]:
        return "ok"

    count_map = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    return count_map.get(total, "unknown")

# ---  Gesture Display Info ---
GESTURE_INFO = {
    "zero":     ("0",        (100, 200, 100)),
    "one":      ("1",        (100, 220, 100)),
    "two":      ("2",        (100, 240, 100)),
    "three":    ("3",        (100, 255, 100)),
    "four":     ("4",        (80,  230, 80)),
    "five":     ("5",        (60,  210, 60)),
    "thumbsup": ("GOOD!",    (0,   220, 255)),
   
    "call me":  ("CALL ME",  (255, 150, 0)),
    "rock on":  ("ROCK ON",  (200, 50,  255)),
    "ok":       ("OK!",      (0,   255, 180)),
    "unknown":  ("...",      (120, 120, 120)),
}

# ---  Main ---
cap = cv2.VideoCapture(0)
for _ in range(30):
    cap.read()

last_announced = None
stable_text = ""
stable_count = 0
THRESHOLD = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "unknown"

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        label   = results.multi_handedness[0].classification[0].label
        lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

        fingers = []
        if label == "Right":
            fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
        else:
            fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)
        for i in range(1, 5):
            fingers.append(1 if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i]-2][1] else 0)

        gesture = detect_gesture(lm_list, fingers, label)

    else:
        
        last_announced = None
        stable_count = 0
        stable_text = ""

    # --- Stability Logic ---
    if gesture == stable_text:
        stable_count += 1
    else:
        stable_count = 0
        stable_text = gesture

    if stable_count == THRESHOLD:
        if stable_text != "unknown" and stable_text != last_announced:
            speak(stable_text.replace("thumbsup", "thumbs up"))
            last_announced = stable_text
        elif stable_text == "unknown":
            last_announced = None

    # --- UI ---
    label_text, color = GESTURE_INFO.get(gesture, ("...", (120,120,120)))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Gesture: {label_text}", (10, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

    bar_w = int((stable_count / THRESHOLD) * 200)
    cv2.rectangle(frame, (10, 65), (210, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 65), (10 + bar_w, 80), color, -1)

    draw_braille_symbol(frame, gesture, w - 70, h - 100)
    cv2.putText(frame, "symbol", (w - 72, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Gesture System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()