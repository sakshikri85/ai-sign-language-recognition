import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import subprocess
import time

# --- ✅ Windows PowerShell TTS — pyttsx3 nahi ---
def speak(text):
    safe_text = text.replace('"', '').replace("'", "")
    subprocess.Popen(
        ['powershell', '-Command',
         f'Add-Type -AssemblyName System.Speech; '
         f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
         f'$s.Speak("{safe_text}")'],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

# --- Braille Dots ---
# Braille cell: 2 columns (col 0=left, col 1=right), 3 rows (row 0,1,2 = top to bottom)
# Standard Braille dot numbering:
#   Dot 1 = (0,0)   Dot 4 = (1,0)
#   Dot 2 = (0,1)   Dot 5 = (1,1)
#   Dot 3 = (0,2)   Dot 6 = (1,2)

BRAILLE_DOTS = {
    # --- Numbers ---
    "0":  [],
    "1":  [(0,0)],
    "2":  [(0,0),(0,1)],
    "3":  [(0,0),(1,0)],
    "4":  [(0,0),(1,0),(1,1)],
    "5":  [(0,0),(1,1)],
    "6":  [(0,0),(1,0),(0,1)],
    "7":  [(0,0),(1,0),(0,1),(1,1)],
    "8":  [(0,0),(0,1),(1,1)],
    "9a": [(1,0),(0,1)],

    # --- Alphabets (Standard Braille Grade 1) ---
    "A":  [(0,0)],
    "B":  [(0,0),(0,1)],
    "C":  [(0,0),(1,0)],
    "D":  [(0,0),(1,0),(1,1)],
    "E":  [(0,0),(1,1)],
    "E1": [(0,0),(1,1),(0,2)],
    "F":  [(0,0),(1,0),(0,1)],
    "G":  [(0,0),(1,0),(0,1),(1,1)],
    "H":  [(0,0),(0,1),(1,1)],
    "I":  [(1,0),(0,1)],
    "J":  [(1,0),(0,1),(1,1)],
    "K":  [(0,0),(0,2)],
    "L":  [(0,0),(0,1),(0,2)],
    "M":  [(0,0),(1,0),(0,2)],
    "N":  [(0,0),(1,0),(1,1),(0,2)],
    "O":  [(0,0),(1,1),(0,2)],
    "P":  [(0,0),(1,0),(0,1),(0,2)],
    "Q":  [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "R":  [(0,0),(0,1),(1,1),(0,2)],
    "S":  [(1,0),(0,1)],
    "T":  [(1,0),(0,1),(1,1)],
    "U":  [(0,0),(0,2),(1,2)],
    "V":  [(0,0),(0,1),(0,2),(1,2)],
    "W":  [(1,0),(0,1),(1,1),(1,2)],
    "X":  [(0,0),(1,0),(0,2),(1,2)],
    "Y":  [(0,0),(1,0),(1,1),(0,2),(1,2)],
    "Z":  [(0,0),(1,0),(0,2)],

    # --- Single Word Gestures ---
    "HELLO_HI":     [(0,0),(0,1),(1,1)],
    "HELP":         [(0,0),(0,1),(0,2)],
    "SORRY":        [(1,0),(0,1),(0,2)],
    "THANK":        [(0,0),(1,0),(0,2)],
    "GOOD":         [(0,0),(1,0),(0,1)],
    "BAD":          [(1,0),(1,1),(1,2)],
    "HAPPY":        [(0,0),(1,0),(0,1),(1,1)],
    "SAD":          [(0,1),(1,1),(0,2)],
    "ANGRY":        [(0,0),(0,1),(0,2),(1,2)],
    "LOVE":         [(0,0),(1,0),(1,2)],
    "STOP":         [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)],
    "YES":          [(0,0),(0,2)],
    "NO":           [(1,0),(1,2)],
    "WANT":         [(0,0),(1,1),(0,2),(1,2)],
    "NEED":         [(0,0),(0,1),(1,2)],
    "WATER":        [(1,0),(0,1),(1,1)],
    "FOOD":         [(0,0),(1,0),(1,1),(1,2)],
    "COME":         [(0,0),(1,1),(0,2)],
    "GO":           [(1,0),(0,2)],
    "KNOW":         [(0,0),(0,1),(1,1),(1,2)],
    "THINK":        [(1,0),(0,1),(0,2)],
    "UNDERSTAND":   [(0,0),(1,0),(0,2),(1,2)],
    "PLEASE":       [(0,1),(1,1),(1,2)],
    "WELCOME":      [(0,0),(1,0),(0,1),(0,2)],
    "FRIEND":       [(1,0),(0,1),(1,2)],
    "BEAUTIFUL":    [(0,0),(1,0),(0,1),(1,2)],
    "TIRED":        [(0,1),(1,1),(0,2),(1,2)],
    "SICK":         [(0,0),(0,2),(1,1)],
    "FINE":         [(0,0),(1,1),(0,1)],
    "SLEEP":        [(1,0),(1,1),(0,2)],
    "COLD":         [(0,0),(0,1),(1,0)],
    "FEVER":        [(1,0),(0,1),(1,1),(0,2)],
    "MEDICINE":     [(0,0),(1,0),(1,1),(0,2),(1,2)],
    "PHONE":        [(1,0),(0,2),(1,2)],
    "TODAY":        [(0,0),(0,1),(1,1)],
    "NOW":          [(1,0),(0,1),(0,2)],
    "WAIT":         [(0,0),(1,2)],

    # --- NEW: Single Word Gestures (missing se add kiye) ---
    "A LOT":        [(0,0),(1,0),(0,1),(0,2),(1,2)],
    "ABUSE":        [(1,0),(0,1),(1,1),(1,2)],
    "AFRAID":       [(0,0),(0,1),(0,2),(1,0)],
    "AGREE":        [(0,0),(1,1),(0,1)],
    "ALL":          [(0,0),(1,0),(0,2)],
    "ANY":          [(1,0),(1,1),(0,2)],
    "ANYTHING":     [(0,0),(1,0),(1,1),(0,2)],
    "APPRECIATE":   [(0,0),(0,1),(1,0),(1,2)],
    "BECOME":       [(0,0),(0,1),(1,1),(0,2)],
    "BED":          [(1,0),(0,1),(0,2)],
    "BORED":        [(0,0),(1,1),(1,2)],
    "BRING":        [(0,0),(1,0),(0,1),(1,2)],
    "CHAT":         [(1,0),(0,1),(1,1),(0,2),(1,2)],
    "CLASS":        [(0,0),(0,1),(1,1),(1,2)],
    "COLLEGE_SCHOOL": [(0,0),(1,0),(0,2),(1,2)],
    "COMB":         [(1,0),(0,2),(1,1)],
    "CONGRATULATIONS": [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "CRYING":       [(0,1),(1,1),(1,2)],
    "DARE":         [(0,0),(0,2),(1,2)],
    "DIFFERENCE":   [(0,0),(1,0),(1,1),(0,2),(1,2)],
    "DILEMMA":      [(1,0),(0,1),(0,2),(1,2)],
    "DISAPPOINTED": [(0,0),(0,1),(0,2)],
    "DO":           [(1,0),(1,1)],
    "DON'T CARE":   [(0,0),(1,1),(0,2)],
    "ENJOY":        [(0,0),(1,0),(0,1),(1,1)],
    "FAVOUR":       [(0,1),(0,2),(1,2)],
    "FREE":         [(0,0),(1,2)],
    "FROM":         [(1,0),(0,1),(0,2)],
    "GLASS":        [(0,0),(1,0),(0,1)],
    "GOT":          [(1,0),(0,1),(1,2)],
    "GRATEFUL":     [(0,0),(0,1),(1,0),(1,1)],
    "HAD":          [(0,0),(0,2),(1,0)],
    "HAPPENED":     [(0,0),(0,1),(1,1),(0,2)],
    "HEAR":         [(0,0),(1,0),(0,2)],
    "HEART":        [(0,0),(1,1),(0,2),(1,2)],
    "HIDING":       [(1,0),(0,2),(1,1),(0,0)],
    "HOW":          [(0,0),(1,0),(1,1),(1,2)],
    "HUNGRY":       [(0,0),(0,1),(0,2),(1,2)],
    "HURT":         [(0,0),(1,0),(0,1),(1,2)],
    "He is going into the room": [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)],
    "KIND":         [(0,0),(1,1),(1,2)],
    "LEAVE":        [(0,0),(0,1),(0,2),(1,0)],
    "LIGHT":        [(1,0),(1,1),(0,2),(1,2)],
    "LIKE":         [(0,0),(1,0),(1,2)],
    "LIKE_LOVE":    [(0,0),(1,0),(0,1),(0,2)],
    "MAKE":         [(0,0),(1,1),(0,1)],
    "MEAN IT":      [(1,0),(0,1),(1,1)],
    "MEET":         [(0,0),(0,2),(1,2)],
    "NAME":         [(0,0),(1,0),(0,1)],
    "NEVER":        [(1,0),(0,1),(1,2)],
    "NICE":         [(0,0),(1,0),(0,2)],
    "NOT":          [(0,0),(0,1),(0,2)],
    "NUMBER":       [(0,0),(1,0),(1,1)],
    "New folder":   [(1,0),(1,1),(1,2)],
    "OLD_AGE":      [(0,0),(0,1),(1,0),(0,2)],
    "ON THE WAY":   [(0,0),(1,0),(1,1),(0,2)],
    "ONWARDS":      [(0,0),(1,0),(0,1),(1,2)],
    "OUTSIDE":      [(0,0),(0,2),(1,1)],
    "PLACE":        [(1,0),(0,1),(0,2),(1,2)],
    "PLANNED":      [(0,0),(1,0),(0,2),(1,2)],
    "POUR":         [(0,1),(1,1),(0,2),(1,2)],
    "PREPARE":      [(0,0),(1,0),(0,1),(0,2),(1,2)],
    "PROMISE":      [(0,0),(0,1),(1,1),(1,2)],
    "REALLY":       [(0,0),(1,0),(0,1),(1,1),(1,2)],
    "REPEAT":       [(0,0),(0,1),(0,2),(1,2)],
    "ROOM":         [(0,0),(1,0),(1,2)],
    "SERVE":        [(1,0),(0,1),(1,1),(0,2)],
    "SHIRT":        [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "SITTING":      [(0,0),(0,2),(1,0),(1,2)],
    "SLOWER":       [(0,1),(1,1),(0,2)],
    "SO MUCH":      [(0,0),(1,0),(0,2),(1,1)],
    "SOFTLY":       [(0,0),(0,1),(1,2)],
    "SOME HOW":     [(0,0),(0,1),(1,0)],
    "SOME ONE":     [(1,0),(0,1),(0,2)],
    "SOMETHING":    [(0,0),(1,1),(1,2)],
    "SPEAK":        [(0,0),(1,0),(1,1),(0,2)],
    "STUBBORN":     [(0,0),(0,1),(1,0),(1,2)],
    "SURE":         [(0,0),(0,2),(1,0)],
    "TAKE CARE":    [(0,0),(1,0),(0,1),(1,2)],
    "TAKE TIME":    [(0,0),(0,1),(1,1),(0,2)],
    "TALK":         [(0,0),(1,0),(0,2)],
    "TELL":         [(0,0),(0,1),(0,2),(1,1)],
    "THAT":         [(1,0),(0,2),(1,2)],
    "THERE":        [(0,0),(1,0),(0,2)],
    "THINGS":       [(0,0),(0,1),(1,0),(0,2)],
    "THIRSTY":      [(1,0),(0,1),(1,1),(1,2)],
    "THIS ONE":     [(0,0),(1,1),(0,2)],
    "TRAIN":        [(0,0),(1,0),(0,1),(1,1)],
    "TRUST":        [(0,0),(0,1),(1,0),(1,1)],
    "TRUTH":        [(0,0),(1,0),(0,2),(1,1)],
    "TURN ON":      [(0,0),(0,1),(0,2),(1,0),(1,2)],
    "This place is beautiful": [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "VERY":         [(0,0),(0,2),(1,2)],
    "WEAR":         [(0,0),(0,1),(1,1),(1,2)],
    "WHAT":         [(0,0),(1,0),(0,2)],
    "WHEN":         [(0,0),(0,1),(1,0),(1,2)],
    "WHERE":        [(1,0),(0,1),(1,1)],
    "WHO":          [(0,0),(0,1),(0,2),(1,1)],
    "WORRY":        [(0,0),(1,0),(1,1),(0,2),(1,2)],
    "YOU":          [(0,0),(1,0),(0,1)],

    # --- Phrase Gestures ---
    "how are you":              [(0,0),(1,0),(0,1)],
    "good morning":             [(0,0),(1,0),(1,1),(0,2)],
    "happy birthday":           [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "i am fine. thank you sir": [(0,0),(0,1),(1,2)],
    "i need help":              [(0,0),(1,1),(0,2)],
    "help me":                  [(0,0),(0,1),(1,0)],
    "i am hungry":              [(1,0),(0,1),(1,2)],
    "i am crying":              [(0,0),(1,1),(1,2)],
    "i am feeling cold":        [(0,0),(1,0),(0,2)],
    "i am feeling bored":       [(1,0),(1,1),(0,2)],
    "i am really grateful":     [(0,0),(0,1),(1,1),(0,2)],
    "where are you from":       [(0,0),(1,0),(1,2)],
    "who are you":              [(1,0),(0,1),(0,2),(1,2)],
    "what you want":            [(0,0),(0,2),(1,1)],
    "speak softly":             [(0,1),(1,1),(0,2)],
    "congratulations":          [(0,0),(1,0),(0,1),(1,1)],
    "keepsmile":                [(0,0),(0,1),(1,2)],

    # --- NEW: All Remaining Phrase Gestures from Labels ---
    "are you free today":           [(0,0),(1,0),(0,1),(0,2)],
    "are you hiding something":     [(1,0),(0,1),(0,2),(1,2)],
    "bring water for me":           [(0,0),(1,0),(0,1),(1,1),(1,2)],
    "can i help you":               [(0,0),(0,1),(0,2),(1,0)],
    "can you repeat that please":   [(0,0),(1,0),(0,2),(1,1)],
    "comb your hair":               [(1,0),(0,1),(1,1),(0,2)],
    "could you please talk slower": [(0,0),(0,1),(1,1),(1,2)],
    "do me a favour":               [(0,0),(1,0),(1,1),(0,2)],
    "do not abuse him":             [(0,0),(0,2),(1,0),(1,2)],
    "do not hurt me":               [(0,0),(1,1),(0,1),(1,2)],
    "do not make me angry":         [(0,0),(0,1),(0,2),(1,2)],
    "do not worry":                 [(0,0),(1,0),(0,1)],
    "do you need something":        [(0,0),(0,1),(1,1),(0,2),(1,2)],
    "go and sleep":                 [(1,0),(1,1),(0,2),(1,2)],
    "had your food":                [(0,0),(1,0),(0,2)],
    "he is on the way":             [(0,0),(0,1),(1,0)],
    "he she is my friend":          [(0,0),(1,0),(1,2)],
    "hi how are you":               [(0,0),(0,1),(0,2)],
    "how can i help you":           [(0,0),(1,1),(0,2),(1,2)],
    "how can i trust you":          [(1,0),(0,1),(1,2)],
    "how dare you":                 [(0,0),(1,0),(0,1),(1,2)],
    "how old are you":              [(0,0),(0,1),(1,1)],
    "i am suffering from fever":    [(0,0),(1,0),(0,1),(1,1)],
    "i am tired":                   [(0,1),(1,1),(0,2)],
    "i am very happy":              [(0,0),(1,0),(0,2),(1,2)],
    "i enjoyed a lot":              [(0,0),(1,1),(0,1)],
    "i got hurt":                   [(1,0),(0,1),(0,2)],
    "i like you i love you":        [(0,0),(1,0),(0,1),(0,2)],
    "i need water":                 [(0,0),(0,1),(1,0),(1,2)],
    "i promise":                    [(0,0),(0,2),(1,1)],
    "it does not make any difference to me": [(0,0),(1,0),(0,1),(1,1),(1,2)],
    "it was nice chatting with you": [(0,0),(0,1),(0,2),(1,0)],
    "let him take time":            [(1,0),(1,1),(1,2)],
    "nice to meet you":             [(0,0),(1,0),(1,1)],
    "prepare the bed":              [(0,0),(0,1),(0,2),(1,2)],
    "serve the food":               [(0,0),(1,0),(0,2)],
    "take care of yourself":        [(0,0),(0,1),(1,1),(0,2)],
    "tell me truth":                [(0,0),(1,0),(0,1),(1,2)],
    "thank you so much":            [(0,0),(0,1),(1,0)],
    "try to understand":            [(1,0),(0,2),(1,2)],
    "turn on light turn off light": [(0,0),(0,1),(0,2),(1,0),(1,2)],
    "we are all with you":          [(0,0),(1,0),(0,1),(1,1),(0,2)],
    "wear the shirt":               [(0,0),(1,1),(0,2)],
    "what are you doing":           [(0,0),(0,1),(1,1)],
    "what do you do":               [(1,0),(0,1),(1,1)],
    "what do you think":            [(0,0),(1,0),(0,2),(1,1)],
    "what do you want to become":   [(0,0),(0,1),(0,2)],
    "what happened":                [(0,0),(1,0),(1,2)],
    "what have you planned for your career": [(0,0),(1,0),(0,1),(0,2),(1,2)],
    "what is your phone number":    [(1,0),(0,1),(0,2),(1,2)],
    "which collegeschool are you from": [(0,0),(1,0),(0,1),(1,1)],
    "why are you angry":            [(0,0),(0,2),(1,0)],
    "why are you crying":           [(0,0),(1,1),(1,2)],
    "why are you disappointed":     [(0,0),(0,1),(1,0),(0,2)],
    "you are bad":                  [(1,0),(1,1),(0,2)],
    "you are good":                 [(0,0),(1,0),(1,1)],
    "you are welcome":              [(0,0),(0,1),(1,2)],
    "you can do it":                [(0,0),(0,1),(0,2),(1,1)],
    "you do anything, i do not care": [(0,0),(1,0),(0,2)],
    "you need a medicine, take this one": [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)],
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
                cv2.circle(frame, (cx, cy), 7, (0, 150, 255), 1)
            else:
                cv2.circle(frame, (cx, cy), 5, (60, 60, 60), -1)


# --- Load Model & Labels ---
model  = tf.keras.models.load_model("gesture_model.h5", compile=False, safe_mode=False)
labels = np.load("labels.npy", allow_pickle=True)

# --- Mediapipe ---
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

IMG_SIZE             = 64
speak_cooldown       = 2.5
speak_time           = 0
last_spoken          = ""
CONFIDENCE_THRESHOLD = 0.75

print("Started — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected")
        break

    frame   = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    now     = time.time()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            x_list = [int(lm.x * w) for lm in handLms.landmark]
            y_list = [int(lm.y * h) for lm in handLms.landmark]

            x_min = max(0, min(x_list) - 20)
            x_max = min(w, max(x_list) + 20)
            y_min = max(0, min(y_list) - 20)
            y_max = min(h, max(y_list) + 20)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            hand_img = hand_img.astype("float32") / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            pred         = model.predict(hand_img, verbose=0)
            class_id     = np.argmax(pred)
            confidence   = float(pred[0][class_id])
            gesture_name = str(labels[class_id])

            if confidence < CONFIDENCE_THRESHOLD:
                gesture_name = "..."

            # Bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Dark panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Gesture text
            cv2.putText(frame, f"{gesture_name}", (10, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

            # Confidence bar
            bar_w     = int(confidence * 200)
            bar_color = (0, 255, 0) if confidence > 0.85 else (0, 200, 255)
            cv2.rectangle(frame, (10, 50), (210, 65), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 50), (10 + bar_w, 65), bar_color, -1)
            cv2.putText(frame, f"{int(confidence*100)}%", (215, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Braille + Voice
            if gesture_name != "...":
                draw_braille_symbol(frame, gesture_name, w - 70, h - 110)
                cv2.putText(frame, "symbol", (w - 72, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

                if (now - speak_time) > speak_cooldown:
                    speak(gesture_name)
                    speak_time = now

    else:
        last_spoken = ""

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()