import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import subprocess
import time
import streamlit as st
from collections import deque

# ===================== CONFIG =====================
MODEL_PATH  = "gesture_model.h5"
LABELS_PATH = "labels.npy"

# ===================== WINDOWS TTS =====================
def speak(text):
    safe_text = str(text).replace('"', '').replace("'", "")
    subprocess.Popen(
        ['powershell', '-Command',
         f'Add-Type -AssemblyName System.Speech; '
         f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
         f'$s.Speak("{safe_text}")'],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

# ===================== LANDMARK EXTRACTION =====================
def extract_landmarks_live(handLms, w, h):
    """
    Same normalisation as landmark_trainer.py — must match exactly.
    Returns 63-dim float32 vector.
    """
    lm  = handLms.landmark
    vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    vec -= vec[0]
    scale = np.max(np.abs(vec)) + 1e-8
    vec  /= scale
    return vec.flatten()

# ===================== BRAILLE DOTS =====================
BRAILLE_DOTS = {
    "0": [], "1": [(0,0)], "2": [(0,0),(0,1)],
    "3": [(0,0),(1,0),(0,1)], "4": [(0,0),(1,0),(0,1),(1,1)],
    "5": [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "6": [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)],
    "7": [(0,0),(0,1),(0,2)], "8": [(1,0),(1,1),(1,2)],
    "9a": [(0,0),(1,2)],
    "A": [(0,0)], "B": [(0,0),(0,1)], "C": [(0,0),(1,0)],
    "D": [(0,0),(1,0),(1,1)], "E": [(0,0),(1,1)],
    "F": [(0,0),(1,0),(0,1)], "G": [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "H": [(0,0),(0,1),(1,1)], "I": [(1,0),(0,1)],
    "J": [(1,0),(0,1),(1,1)], "K": [(0,0),(1,0),(0,2)],
    "L": [(0,0),(0,1),(0,2)], "M": [(0,0),(1,0),(0,1),(1,1)],
    "N": [(0,0),(1,0),(1,1),(0,2)], "O": [(0,0),(1,0),(1,2)],
    "P": [(0,0),(1,0),(0,1),(0,2)], "Q": [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "R": [(0,0),(0,1),(1,1),(0,2)], "S": [(0,0),(0,1),(0,2),(1,2)],
    "T": [(0,0),(1,0),(0,1),(0,2)], "U": [(0,0),(0,2),(1,2)],
    "V": [(0,0),(0,1),(0,2),(1,2)], "W": [(1,0),(0,1),(1,1),(1,2)],
    "X": [(0,0),(1,1),(0,2)], "Y": [(0,0),(1,0),(1,1),(1,2)],
    "Z": [(0,0),(1,0),(0,2),(1,2)],
    "HELLO_HI": [(0,0),(1,0),(1,1)], "HELP": [(0,0),(0,1),(1,2)],
    "SORRY": [(0,0),(1,1),(1,2)], "THANK": [(0,0),(1,0),(0,2)],
    "GOOD": [(0,0),(1,0),(0,1)], "BAD": [(1,0),(1,1),(1,2)],
    "HAPPY": [(0,0),(1,0),(0,1),(1,1)], "SAD": [(0,1),(1,1),(0,2)],
    "ANGRY": [(0,0),(0,1),(0,2),(1,2)], "LOVE": [(0,0),(1,0),(1,2)],
    "STOP": [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)],
    "YES": [(0,0),(0,2)], "NO": [(1,0),(1,2)],
    "WATER": [(1,0),(0,1),(1,1)], "FOOD": [(0,0),(1,0),(1,1),(1,2)],
    "PLEASE": [(0,1),(1,1),(1,2)], "WELCOME": [(0,0),(1,0),(0,1),(0,2)],
}

def draw_braille_symbol(frame, gesture_name, x, y):
    dots = BRAILLE_DOTS.get(gesture_name, [(0,0),(1,2)])
    cv2.rectangle(frame, (x-5, y-5), (x+45, y+75), (30, 30, 30), -1)
    cv2.rectangle(frame, (x-5, y-5), (x+45, y+75), (100, 100, 100), 1)
    for col in range(2):
        for row in range(3):
            cx = x + col * 20 + 10
            cy = y + row * 22 + 10
            if (col, row) in dots:
                cv2.circle(frame, (cx, cy), 7, (0, 100, 255), -1)
            else:
                cv2.circle(frame, (cx, cy), 5, (60, 60, 60), -1)

# ===================== FINGER GESTURE =====================
tip_ids = [4, 8, 12, 16, 20]

def detect_finger_gesture(lm_list, fingers, label):
    total = sum(fingers)
    if fingers[0] and not any(fingers[1:]):
        if lm_list[4][1] < lm_list[3][1] < lm_list[2][1]:
            return "thumbsup"
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4] and not fingers[0]:
        return "victory"
    if fingers[0] and fingers[4] and not fingers[1] and not fingers[2] and not fingers[3]:
        return "call me"
    if fingers[1] and fingers[4] and not fingers[2] and not fingers[3]:
        return "rock on"
    dist = np.hypot(lm_list[4][0]-lm_list[8][0], lm_list[4][1]-lm_list[8][1])
    if dist < 40 and fingers[2] and fingers[3] and fingers[4]:
        return "ok"
    return {0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five"}.get(total, "unknown")

# ===================== MOTION GESTURE =====================
HISTORY_LEN=12; ZOOM_HISTORY_LEN=8; MOTION_THRESHOLD=40
VERT_THRESHOLD=35; ZOOM_THRESHOLD_RATIO=0.12; WAVE_SPEED=18
PUSH_PULL_THRESHOLD=25; PALM_IDX=9; THUMB_TIP_IDX=4; INDEX_TIP_IDX=8

def calc_dist(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def detect_circle(pos_hist):
    if len(pos_hist) < HISTORY_LEN: return None
    pts = list(pos_hist)
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    if max(xs)-min(xs)<40 or max(ys)-min(ys)<40: return None
    cross=0
    for i in range(1, len(pts)-1):
        dx1=pts[i][0]-pts[i-1][0]; dy1=pts[i][1]-pts[i-1][1]
        dx2=pts[i+1][0]-pts[i][0]; dy2=pts[i+1][1]-pts[i][1]
        cross+=dx1*dy2-dy1*dx2
    return ("Circle Clockwise" if cross>0 else "Circle Counter") if abs(cross)>500 else None

def detect_tilt(pos_hist):
    if len(pos_hist)<HISTORY_LEN: return None
    pts=list(pos_hist); half=len(pts)//2
    dy=np.mean([p[1] for p in pts[half:]])-np.mean([p[1] for p in pts[:half]])
    dx=np.mean([p[0] for p in pts[half:]])-np.mean([p[0] for p in pts[:half]])
    if abs(dx)>30 and abs(dy)>30:
        if dx>0 and dy>0: return "Tilt Right"
        elif dx<0 and dy>0: return "Tilt Left"
    return None

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="AI Sign Language Recognition", layout="wide")
st.title("AI Powered Sign Language Recognition and Communication System")
st.markdown("---")

st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "Detection Mode",
    ["AI Model (Sign Language)", "Finger Gesture", "Motion Gesture"],
)
voice_on       = st.sidebar.checkbox("🔊 Voice Output", value=True)
speak_cooldown = st.sidebar.slider("Voice Cooldown (sec)", 1.0, 5.0, 2.5, 0.5)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.3, 0.95, 0.60, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model type:** Landmark MLP")

# ===================== LOAD MODEL =====================
model       = None
class_names = []
if "AI Model" in mode:
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        if os.path.exists(LABELS_PATH):
            class_names = list(np.load(LABELS_PATH, allow_pickle=True))
        st.sidebar.success(f"✅ Model loaded! ({len(class_names)} classes)")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Model not found: {e}")
        class_names = ["Hand Detected"]

# Start/Stop
col1, col2 = st.columns(2)
with col1: start = st.button("▶️ Start Recognition", use_container_width=True)
with col2: stop  = st.button("⏹️ Stop Recognition",  use_container_width=True)

if "run" not in st.session_state: st.session_state.run = False
if start: st.session_state.run = True
if stop:  st.session_state.run = False

frame_window   = st.empty()
ic1, ic2, ic3  = st.columns(3)
gesture_box    = ic1.empty()
confidence_box = ic2.empty()
sentence_box   = ic3.empty()

# ===================== MAIN LOOP =====================
if st.session_state.run:
    mp_hands_sol = mp.solutions.hands
    mp_draw      = mp.solutions.drawing_utils
    hands = mp_hands_sol.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    for _ in range(15): cap.read()

    speak_time = 0
    sentence   = ""

    # Per-hand state dicts
    last_spoken  = {}
    stable_text  = {}
    stable_count = {}
    pos_hist     = {}
    dist_hist    = {}
    z_hist       = {}
    last_mot_lbl = {}
    last_mot_t   = {}
    STABLE_THR   = 12

    def init_hand(idx):
        if idx not in stable_text:
            stable_text[idx]=stable_count[idx]=0
            stable_text[idx]=""
            last_spoken[idx]=""
        if idx not in pos_hist:
            pos_hist[idx]=deque(maxlen=HISTORY_LEN)
            dist_hist[idx]=deque(maxlen=ZOOM_HISTORY_LEN)
            z_hist[idx]=deque(maxlen=HISTORY_LEN)
            last_mot_lbl[idx]=""
            last_mot_t[idx]=0

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected!")
            break

        frame   = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now     = time.time()

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,80), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        display_gesture    = "..."
        display_confidence = 0.0
        seen_hands         = set()

        if results.multi_hand_landmarks:
            num_h = len(results.multi_hand_landmarks)
            cv2.putText(frame, f"Hands: {num_h}", (w-160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)

            for hand_idx, (handLms, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):

                seen_hands.add(hand_idx)
                init_hand(hand_idx)
                label_hand = handedness.classification[0].label

                mp_draw.draw_landmarks(frame, handLms, mp_hands_sol.HAND_CONNECTIONS)

                x_list=[int(lm.x*w) for lm in handLms.landmark]
                y_list=[int(lm.y*h) for lm in handLms.landmark]
                x_min=max(0,min(x_list)-20); x_max=min(w,max(x_list)+20)
                y_min=max(0,min(y_list)-20); y_max=min(h,max(y_list)+20)

                box_color = (0,255,0) if hand_idx==0 else (255,165,0)
                cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),box_color,2)
                cv2.putText(frame, f"Hand {hand_idx+1} ({label_hand})",
                            (x_min, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                lm_list=[(int(lm.x*w),int(lm.y*h)) for lm in handLms.landmark]
                gesture_name = "..."
                confidence   = 0.0

                # ======= AI MODEL (Landmark-based) =======
                if "AI Model" in mode:
                    if model is not None:
                        feat  = extract_landmarks_live(handLms, w, h)
                        feat  = np.expand_dims(feat, axis=0)   # (1,63)
                        preds = model.predict(feat, verbose=0)[0]
                        idx   = np.argmax(preds)
                        confidence   = float(preds[idx])
                        gesture_name = (str(class_names[idx])
                                        if confidence >= conf_threshold else "...")
                    else:
                        gesture_name="Hand Detected"; confidence=1.0

                    if gesture_name != "...":
                        if gesture_name != last_spoken[hand_idx]:
                            sentence += gesture_name + " "
                            last_spoken[hand_idx] = gesture_name
                        if voice_on and (now-speak_time) > speak_cooldown:
                            speak(gesture_name); speak_time=now

                # ======= FINGER GESTURE =======
                elif "Finger" in mode:
                    fingers=[]
                    if label_hand=="Right":
                        fingers.append(1 if lm_list[4][0]>lm_list[3][0] else 0)
                    else:
                        fingers.append(1 if lm_list[4][0]<lm_list[3][0] else 0)
                    for i in range(1,5):
                        fingers.append(1 if lm_list[tip_ids[i]][1]<lm_list[tip_ids[i]-2][1] else 0)

                    detected=detect_finger_gesture(lm_list,fingers,label_hand)
                    confidence=1.0
                    if detected==stable_text[hand_idx]: stable_count[hand_idx]+=1
                    else: stable_count[hand_idx]=0; stable_text[hand_idx]=detected

                    if stable_count[hand_idx]>=STABLE_THR:
                        gesture_name=stable_text[hand_idx] if stable_text[hand_idx]!="unknown" else "..."
                        if gesture_name!="..." and voice_on and (now-speak_time)>speak_cooldown:
                            speak(gesture_name); speak_time=now

                    bar_x=10+hand_idx*170; bar_y=55
                    bar_w=int((min(stable_count[hand_idx],STABLE_THR)/STABLE_THR)*150)
                    cv2.rectangle(frame,(bar_x,bar_y),(bar_x+150,bar_y+12),(50,50,50),-1)
                    cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_w,bar_y+12),(0,255,100),-1)

                # ======= MOTION GESTURE =======
                elif "Motion" in mode:
                    lm3=[(int(pt.x*w),int(pt.y*h),int(pt.z*1000)) for pt in handLms.landmark]
                    cx,cy,cz=lm3[PALM_IDX]
                    ph=pos_hist[hand_idx]; dh=dist_hist[hand_idx]; zh=z_hist[hand_idx]
                    ph.append((cx,cy)); zh.append(cz)
                    dh.append(calc_dist(
                        (lm3[THUMB_TIP_IDX][0],lm3[THUMB_TIP_IDX][1]),
                        (lm3[INDEX_TIP_IDX][0],lm3[INDEX_TIP_IDX][1])
                    ))
                    det=None
                    if len(ph)==HISTORY_LEN:
                        dx=ph[-1][0]-ph[0][0]; dy=ph[-1][1]-ph[0][1]
                        if abs(dx)>abs(dy) and abs(dx)>MOTION_THRESHOLD:
                            det="Move Right" if dx>0 else "Move Left"; ph.clear()
                        elif abs(dy)>abs(dx) and abs(dy)>VERT_THRESHOLD:
                            det="Move Down" if dy>0 else "Move Up"; ph.clear()
                    if not det and len(dh)==ZOOM_HISTORY_LEN:
                        s,e=dh[0],dh[-1]
                        if s!=0:
                            r=(e-s)/s
                            if r>ZOOM_THRESHOLD_RATIO: det="Zoom Out"; dh.clear()
                            elif r<-ZOOM_THRESHOLD_RATIO: det="Zoom In"; dh.clear()
                    if not det and len(zh)==HISTORY_LEN:
                        dz=zh[-1]-zh[0]
                        if dz<-PUSH_PULL_THRESHOLD: det="Push Forward"; zh.clear()
                        elif dz>PUSH_PULL_THRESHOLD: det="Pull Backward"; zh.clear()
                    if not det and len(ph)==HISTORY_LEN:
                        xs=[p[0] for p in ph]
                        if max(xs)-min(xs)>WAVE_SPEED*2: det="Wave Gesture"
                    if not det:
                        c=detect_circle(ph)
                        if c: det=c; ph.clear()
                    if not det:
                        t=detect_tilt(ph)
                        if t: det=t
                    if det:
                        gesture_name=det; last_mot_lbl[hand_idx]=det
                        last_mot_t[hand_idx]=now; confidence=1.0
                        if voice_on and (now-speak_time)>speak_cooldown:
                            speak(det); speak_time=now
                    elif now-last_mot_t[hand_idx]<2.0:
                        gesture_name=last_mot_lbl[hand_idx]; confidence=1.0

                # ---- Draw per-hand label ----
                lcolor=(0,255,100) if gesture_name!="..." else (100,100,100)
                cv2.putText(frame, f"H{hand_idx+1}: {gesture_name}",
                            (x_min, y_max+22), cv2.FONT_HERSHEY_DUPLEX, 0.75, lcolor, 2)

                if gesture_name!="..." and "AI Model" in mode:
                    bw=int(confidence*180)
                    bc=(0,255,0) if confidence>0.85 else (0,200,255)
                    cv2.rectangle(frame,(x_min,y_max+28),(x_min+180,y_max+40),(50,50,50),-1)
                    cv2.rectangle(frame,(x_min,y_max+28),(x_min+bw,y_max+40),bc,-1)
                    cv2.putText(frame,f"{int(confidence*100)}%",
                                (x_min+185,y_max+40),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)

                if gesture_name!="...":
                    draw_braille_symbol(frame, gesture_name, w-70-hand_idx*80, h-110)

                display_gesture=gesture_name; display_confidence=confidence

            # Reset disappeared hands
            for gone in list(pos_hist.keys()):
                if gone not in seen_hands:
                    last_spoken[gone]=""; stable_text[gone]=""
                    stable_count[gone]=0
                    pos_hist[gone].clear(); z_hist[gone].clear(); dist_hist[gone].clear()
        else:
            last_spoken.clear(); stable_text.clear(); stable_count.clear()
            for k in list(pos_hist.keys()):
                pos_hist[k].clear(); z_hist[k].clear(); dist_hist[k].clear()
            cv2.putText(frame,"No hand detected",(10,45),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(80,80,80),2)

        frame_window.image(frame[:,:,::-1], channels="RGB", use_container_width="stretch")
        gesture_box.metric("Gesture", display_gesture)
        confidence_box.metric("Confidence", f"{int(display_confidence*100)}%")
        sentence_box.metric("Sentence", sentence[-40:] if sentence else "—")

    cap.release()
    hands.close()
    st.success("✅ Recognition stopped.")
    if sentence.strip():
        st.info(f"📝 Full sentence: **{sentence.strip()}**")

else:
    st.info("Press **▶️ Start Recognition** to begin.")
    st.markdown("""
    ### Modes Guide
    | Mode | Description |
    |---|---|
    | AI Model | Landmark-based MLP — fast & accurate |
    | Finger Gesture | Finger count / special gestures |
    | Motion Gesture | Hand movement detection |
    """)