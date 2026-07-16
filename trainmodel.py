'''import os
import cv2
import numpy as np
import mediapipe as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
DATASET_PATH = "dataset"
MODEL_PATH   = "gesture_model.h5"
LABELS_PATH  = "labels.npy"
DATA_CACHE   = "landmark_data.npz"  # cache so extraction runs only once

EPOCHS     = 150
BATCH_SIZE = 64
VAL_SPLIT  = 0.20

# ===================== STEP 1: EXTRACT LANDMARKS =====================
mp_hands  = mp.solutions.hands
hands_det = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

def extract_landmarks(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = hands_det.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm  = result.multi_hand_landmarks[0].landmark
    vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)
    vec -= vec[0]           # normalise relative to wrist
    scale = np.max(np.abs(vec)) + 1e-8
    vec  /= scale
    return vec.flatten()    # (63,)

if os.path.exists(DATA_CACHE):
    print(f"Loading cached landmark data from {DATA_CACHE} ...")
    cache = np.load(DATA_CACHE, allow_pickle=True)
    X     = cache["X"]
    y_raw = cache["y"]
    print(f"Loaded {len(X)} samples, {len(np.unique(y_raw))} classes")
else:
    print("Extracting landmarks from dataset (runs once, cached after)...")
    X, y_raw = [], []
    classes  = sorted(os.listdir(DATASET_PATH))
    total    = len(classes)

    for i, cls_name in enumerate(classes):
        cls_dir = os.path.join(DATASET_PATH, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        imgs  = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        count = 0
        for fname in imgs:
            vec = extract_landmarks(os.path.join(cls_dir, fname))
            if vec is not None:
                X.append(vec)
                y_raw.append(cls_name)
                count += 1
        print(f"  [{i+1}/{total}] {cls_name:<40} {count}/{len(imgs)}")

    X     = np.array(X, dtype=np.float32)
    y_raw = np.array(y_raw)
    np.savez_compressed(DATA_CACHE, X=X, y=y_raw)
    print(f"\nSaved cache: {DATA_CACHE}")

hands_det.close()

# ===================== STEP 2: ENCODE LABELS =====================
le      = LabelEncoder()
y_enc   = le.fit_transform(y_raw)
classes = le.classes_
np.save(LABELS_PATH, classes)
num_classes = len(classes)
print(f"\nTotal samples  : {len(X)}")
print(f"Total classes  : {num_classes}")
print(f"Feature dims   : {X.shape[1]}")
print(f"Labels saved   : {LABELS_PATH}")

# ===================== STEP 3: SPLIT =====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=VAL_SPLIT, random_state=42, stratify=y_enc
)
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

# ===================== STEP 4: CLASS WEIGHTS =====================
present = np.unique(y_train)
weights = compute_class_weight('balanced', classes=present, y=y_train)
cw_dict = {i: 1.0 for i in range(num_classes)}
cw_dict.update(dict(zip(present, weights)))

# ===================== STEP 5: MODEL =====================
model = Sequential([
    Dense(512, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ===================== STEP 6: TRAIN =====================
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                             save_best_only=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                               patience=6, min_lr=1e-7, verbose=1)

print(f"\nTraining on {len(X_train)} samples | {num_classes} classes...")
history = model.fit(
    X_train, y_train_oh,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_oh),
    class_weight=cw_dict,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

model.save(MODEL_PATH)

# ===================== STEP 7: PLOT =====================
acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(f"Landmark MLP — {num_classes} classes | Best val acc: {max(val_acc)*100:.1f}%")
axes[0].plot(acc, label='Train'); axes[0].plot(val_acc, label='Val')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_plot.png", dpi=150)
plt.show()

print(f"\n{'='*45}")
print(f"  Best Val Accuracy : {max(val_acc)*100:.1f}%")
print(f"  Model saved       : {MODEL_PATH}")
print(f"  Labels saved      : {LABELS_PATH}")
print(f"{'='*45}")'''



import os
import cv2
import numpy as np
import mediapipe as mp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
DATASET_PATH = "dataset"
MODEL_PATH   = "gesture_model.h5"
LABELS_PATH  = "labels.npy"
DATA_CACHE   = "landmark_data.npz"   # delete this file to re-extract

EPOCHS     = 150    # was 80 — more epochs since cache makes rerun fast
BATCH_SIZE = 64
VAL_SPLIT  = 0.20

# ===================== CLASS MERGE MAP =====================
# Duplicate/overlapping classes merged → one label
# Value = None  →  skip class entirely (garbage / video-only gestures)
MERGE_MAP = {
    # HELP variants
    "help me"                      : "HELP",
    "i need help"                  : "HELP",
    "can i help you"               : "HELP",
    "how can i help you"           : "HELP",
    # HUNGRY
    "i am hungry"                  : "HUNGRY",
    # FEVER
    "i am suffering from fever"    : "FEVER",
    # HAPPY
    "i am very happy"              : "HAPPY",
    "happy birthday"               : "HAPPY",
    # GOOD
    "you are good"                 : "GOOD",
    "good morning"                 : "GOOD",
    # WATER
    "i need water"                 : "WATER",
    "bring water for me"           : "WATER",
    # ANGRY
    "why are you angry"            : "ANGRY",
    "do not make me angry"         : "ANGRY",
    # FOOD
    "had your food"                : "FOOD",
    "serve the food"               : "FOOD",
    # LIKE_LOVE
    "i like you i love you"        : "LIKE_LOVE",
    # THANK
    "thank you so much"            : "THANK",
    "i am fine. thank you sir"     : "THANK",
    # FRIEND
    "he she is my friend"          : "FRIEND",
    # UNDERSTAND
    "try to understand"            : "UNDERSTAND",
    # TRUST
    "how can i trust you"          : "TRUST",
    # TIRED
    "i am tired"                   : "TIRED",
    # PROMISE
    "i promise"                    : "PROMISE",
    # WORRY
    "do not worry"                 : "WORRY",
    # HELLO
    "hi how are you"               : "HELLO_HI",
    # MEET
    "nice to meet you"             : "MEET",
    # SPEAK / SLOWER
    "speak softly"                 : "SPEAK",
    "could you please talk slower" : "SPEAK",
    # REPEAT
    "can you repeat that please"   : "REPEAT",
    # SLEEP
    "go and sleep"                 : "SLEEP",
    # COMB
    "comb your hair"               : "COMB",
    # BAD
    "you are bad"                  : "BAD",
    # WELCOME
    "you are welcome"              : "WELCOME",
    # TRUTH
    "tell me truth"                : "TRUTH",
    # TAKE CARE
    "take care of yourself"        : "TAKE CARE",
    # SORRY
    "do not hurt me"               : "SORRY",
    # ABUSE
    "do not abuse him"             : "ABUSE",
    # --- GARBAGE / VIDEO-ONLY: skip ---
    "New folder"                   : None,
    "He is going into the room"    : None,
    "This place is beautiful"      : None,
}

# ===================== STEP 1: EXTRACT LANDMARKS =====================
mp_hands  = mp.solutions.hands
hands_det = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

def extract_landmarks(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = hands_det.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm  = result.multi_hand_landmarks[0].landmark
    vec = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    vec -= vec[0]
    scale = np.max(np.abs(vec)) + 1e-8
    vec  /= scale
    return vec.flatten()   # (63,)

if os.path.exists(DATA_CACHE):
    print(f"Loading cached landmark data from {DATA_CACHE} ...")
    cache = np.load(DATA_CACHE, allow_pickle=True)
    X     = cache["X"]
    y_raw = cache["y"]
    print(f"Loaded {len(X)} samples, {len(np.unique(y_raw))} classes")

    # Apply merge map to cached labels too
    print("Applying class merge map to cached data...")
    keep_mask = []
    y_merged  = []
    for label in y_raw:
        label_str = str(label)
        if label_str in MERGE_MAP:
            merged = MERGE_MAP[label_str]
            if merged is None:
                keep_mask.append(False)
            else:
                keep_mask.append(True)
                y_merged.append(merged)
        else:
            keep_mask.append(True)
            y_merged.append(label_str)
    keep_mask = np.array(keep_mask)
    X     = X[keep_mask]
    y_raw = np.array(y_merged)
    print(f"After merge: {len(X)} samples, {len(np.unique(y_raw))} classes")

else:
    print("Extracting landmarks from dataset (runs once, then cached)...")
    X, y_raw = [], []
    classes  = sorted(os.listdir(DATASET_PATH))
    total    = len(classes)

    for i, cls_name in enumerate(classes):
        cls_dir = os.path.join(DATASET_PATH, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        # Apply merge map
        if cls_name in MERGE_MAP:
            merged = MERGE_MAP[cls_name]
            if merged is None:
                print(f"  [{i+1}/{total}] SKIPPED (garbage): {cls_name}")
                continue
            effective_label = merged
        else:
            effective_label = cls_name

        imgs  = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        count = 0
        for fname in imgs:
            vec = extract_landmarks(os.path.join(cls_dir, fname))
            if vec is not None:
                X.append(vec)
                y_raw.append(effective_label)
                count += 1
        print(f"  [{i+1}/{total}] {cls_name:<45} → {effective_label:<20} {count}/{len(imgs)}")

    X     = np.array(X, dtype=np.float32)
    y_raw = np.array(y_raw)
    np.savez_compressed(DATA_CACHE, X=X, y=y_raw)
    print(f"\nSaved cache: {DATA_CACHE}")

hands_det.close()

# ===================== STEP 2: ENCODE LABELS =====================
le          = LabelEncoder()
y_enc       = le.fit_transform(y_raw)
classes     = le.classes_
num_classes = len(classes)
np.save(LABELS_PATH, classes)

print(f"\nTotal samples  : {len(X)}")
print(f"Total classes  : {num_classes}  (was 215, now merged)")
print(f"Feature dims   : {X.shape[1]}")
print(f"Labels saved   : {LABELS_PATH}")

# ===================== STEP 3: SPLIT =====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=VAL_SPLIT, random_state=42, stratify=y_enc
)
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)
print(f"Train: {len(X_train)}   Val: {len(X_val)}")

# ===================== STEP 4: CLASS WEIGHTS =====================
present = np.unique(y_train)
weights = compute_class_weight('balanced', classes=present, y=y_train)
cw_dict = {i: 1.0 for i in range(num_classes)}
cw_dict.update(dict(zip(present, weights)))
print(f"Class weight range: {weights.min():.3f} – {weights.max():.3f}")

# ===================== STEP 5: MODEL (deeper MLP) =====================
model = Sequential([
    Dense(1024, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ===================== STEP 6: TRAIN =====================
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                             save_best_only=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                               patience=8, min_lr=1e-7, verbose=1)

print(f"\nTraining on {len(X_train)} samples | {num_classes} classes | {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train_oh,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_oh),
    class_weight=cw_dict,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

model.save(MODEL_PATH)

# ===================== STEP 7: PLOT =====================
acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(f"Landmark MLP — {num_classes} classes | Best val acc: {max(val_acc)*100:.1f}%")
axes[0].plot(acc, label='Train'); axes[0].plot(val_acc, label='Val')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_plot.png", dpi=150)
plt.show()

print(f"\n{'='*45}")
print(f"  Best Val Accuracy : {max(val_acc)*100:.1f}%")
print(f"  Total Classes     : {num_classes}")
print(f"  Model saved       : {MODEL_PATH}")
print(f"{'='*45}")