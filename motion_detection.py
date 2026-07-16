import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
from collections import deque

# --- Windows PowerShell TTS ---
def speak(text):
    subprocess.Popen(
        ['powershell', '-Command',
         f'Add-Type -AssemblyName System.Speech; '
         f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
         f'$s.Speak("{text}")'],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

# --- Braille Dots ---
BRAILLE_DOTS = {
    "Move Right":       [(1,0),(1,1),(1,2)],
    "Move Left":        [(0,0),(0,1),(0,2)],
    "Move Up":          [(0,0),(1,0)],
    "Move Down":        [(0,2),(1,2)],
    "Zoom In":          [(0,1),(1,1),(0,2),(1,2)],
    "Zoom Out":         [(0,0),(1,0),(0,1),(1,1)],
    "Push Forward":     [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)],
    "Pull Backward":    [(0,1),(1,0),(1,2)],
    "Wave Gesture":     [(0,0),(1,1),(0,2)],
    "Circle Clockwise": [(0,0),(1,0),(1,1),(0,2)],
    "Circle Counter":   [(1,0),(0,1),(1,2),(0,0)],
    "Shake Hand":       [(0,0),(1,0),(0,1),(1,1)],
    "Tilt Left":        [(0,0),(0,1),(0,2),(1,2)],
    "Tilt Right":       [(1,0),(1,1),(0,2),(1,2)],
}

def draw_braille_symbol(frame, gesture_name, x, y):
    dots = BRAILLE_DOTS.get(gesture_name, [])
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

# --- Config ---
CAM_INDEX            = 0
HOLD_TIME            = 30
HISTORY_LEN          = 12
ZOOM_HISTORY_LEN     = 8
MOTION_THRESHOLD     = 40
VERT_THRESHOLD       = 35
ZOOM_THRESHOLD_RATIO = 0.12
WAVE_SPEED           = 18
PUSH_PULL_THRESHOLD  = 25
SPEAK_COOLDOWN       = 2.0
PALM_IDX             = 9
THUMB_TIP            = 4
INDEX_TIP            = 8

# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.6,
                          min_tracking_confidence=0.6)
mp_draw  = mp.solutions.drawing_utils

pos_hist  = deque(maxlen=HISTORY_LEN)
dist_hist = deque(maxlen=ZOOM_HISTORY_LEN)
z_hist    = deque(maxlen=HISTORY_LEN)

last_label = ""
last_time  = 0
speak_time = 0

def calc_dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def detect_circle(pos_hist):
    if len(pos_hist) < HISTORY_LEN:
        return None
    pts = list(pos_hist)
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    if max(xs) - min(xs) < 40 or max(ys) - min(ys) < 40:
        return None
    cross = 0
    for i in range(1, len(pts)-1):
        dx1 = pts[i][0]   - pts[i-1][0]
        dy1 = pts[i][1]   - pts[i-1][1]
        dx2 = pts[i+1][0] - pts[i][0]
        dy2 = pts[i+1][1] - pts[i][1]
        cross += dx1 * dy2 - dy1 * dx2
    if abs(cross) > 500:
        return "Circle Clockwise" if cross > 0 else "Circle Counter"
    return None

def detect_tilt(pos_hist):
    if len(pos_hist) < HISTORY_LEN:
        return None
    pts  = list(pos_hist)
    half = len(pts) // 2
    avg_y_first  = np.mean([p[1] for p in pts[:half]])
    avg_y_second = np.mean([p[1] for p in pts[half:]])
    avg_x_first  = np.mean([p[0] for p in pts[:half]])
    avg_x_second = np.mean([p[0] for p in pts[half:]])
    dy = avg_y_second - avg_y_first
    dx = avg_x_second - avg_x_first
    if abs(dx) > 30 and abs(dy) > 30:
        if dx > 0 and dy > 0:
            return "Tilt Right"
        elif dx < 0 and dy > 0:
            return "Tilt Left"
    return None

# --- Camera ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Camera not detected!")
    exit()

for _ in range(30):
    cap.read()

print("Started — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    h, w, _  = frame.shape
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results  = hands.process(rgb)
    detected = None

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        lm = [(int(pt.x*w), int(pt.y*h), int(pt.z*1000)) for pt in hand.landmark]
        cx, cy, cz = lm[PALM_IDX]

        pos_hist.append((cx, cy))
        z_hist.append(cz)
        dist_hist.append(calc_dist(
            (lm[THUMB_TIP][0], lm[THUMB_TIP][1]),
            (lm[INDEX_TIP][0], lm[INDEX_TIP][1])
        ))

        # --- Move Left / Right / Up / Down ---
        if len(pos_hist) == HISTORY_LEN:
            dx = pos_hist[-1][0] - pos_hist[0][0]
            dy = pos_hist[-1][1] - pos_hist[0][1]
            if abs(dx) > abs(dy) and abs(dx) > MOTION_THRESHOLD:
                detected = "Move Right" if dx > 0 else "Move Left"
                pos_hist.clear()
            elif abs(dy) > abs(dx) and abs(dy) > VERT_THRESHOLD:
                detected = "Move Down" if dy > 0 else "Move Up"
                pos_hist.clear()

        # --- Zoom ---
        if not detected and len(dist_hist) == ZOOM_HISTORY_LEN:
            s, e = dist_hist[0], dist_hist[-1]
            if s != 0:
                ratio = (e - s) / s
                if ratio > ZOOM_THRESHOLD_RATIO:
                    detected = "Zoom Out"
                    dist_hist.clear()
                elif ratio < -ZOOM_THRESHOLD_RATIO:
                    detected = "Zoom In"
                    dist_hist.clear()

        # --- Push / Pull ---
        if not detected and len(z_hist) == HISTORY_LEN:
            dz = z_hist[-1] - z_hist[0]
            if dz < -PUSH_PULL_THRESHOLD:
                detected = "Push Forward"
                z_hist.clear()
            elif dz > PUSH_PULL_THRESHOLD:
                detected = "Pull Backward"
                z_hist.clear()

        # --- Wave ---
        if not detected and len(pos_hist) == HISTORY_LEN:
            xs = [p[0] for p in pos_hist]
            if (max(xs) - min(xs)) > WAVE_SPEED * 2:
                detected = "Wave Gesture"

        # --- Circle ---
        if not detected:
            circle = detect_circle(pos_hist)
            if circle:
                detected = circle
                pos_hist.clear()

        # --- Tilt ---
        if not detected:
            tilt = detect_tilt(pos_hist)
            if tilt:
                detected = tilt

    else:
      
        pos_hist.clear()
        z_hist.clear()
        dist_hist.clear()

    now = time.time()

    if detected:
        last_label = detected
        last_time  = now
        
        if (now - speak_time) > SPEAK_COOLDOWN:
            speak(detected)
            speak_time = now

    # --- UI Display ---
    if now - last_time <= HOLD_TIME and last_label:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, last_label, (10, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2)
        draw_braille_symbol(frame, last_label, w-70, h-100)
        cv2.putText(frame, "symbol", (w-72, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Motion Gestures", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")