import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import pyttsx3
import threading

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Camera warm-up
for _ in range(30):
    cap.read()

DETECTION_THRESHOLD = 5
NO_HAND_THRESHOLD = 15
COOLDOWN_FRAMES = 60  # ~2 second

detection_counter = 0
no_hand_counter = 0
cooldown_counter = 0
is_speaking = False  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    if hand_detected:
        no_hand_counter = 0

        if cooldown_counter > 0:
            cooldown_counter -= 1

        detection_counter += 1

        
        if detection_counter >= DETECTION_THRESHOLD and cooldown_counter == 0 and not is_speaking:
            is_speaking = True
            t = threading.Thread(target=lambda: [speak("hello"), setattr(threading.current_thread(), '_spoken', True)])
            t.start()
            detection_counter = 0
            cooldown_counter = COOLDOWN_FRAMES

            
            def reset_speaking(thread):
                thread.join()
                global is_speaking
                is_speaking = False

            threading.Thread(target=reset_speaking, args=(t,)).start()

    else:
        no_hand_counter += 1
        detection_counter = 0

        if no_hand_counter >= NO_HAND_THRESHOLD:
            cooldown_counter = 0
            no_hand_counter = 0

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()