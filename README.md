# AI Sign Language Recognition System

An AI-powered **Sign Language Recognition System** that uses computer vision and deep learning to recognize hand gestures in real time and convert them into **text and audio**, helping improve communication accessibility.

> 🎯 **Objective:** Bridge the communication gap between sign-language users and non-sign-language users using AI and computer vision.

## 🚀 Key Features

* **Real-Time Gesture Recognition** — Detects and recognizes hand gestures through a webcam.
* **Hand Landmark Detection** — Uses MediaPipe to identify 21 hand landmarks for gesture analysis.
* **Deep Learning Classification** — Uses a CNN-based model to classify sign language gestures.
* **Text Conversion** — Converts recognized gestures into corresponding text.
* **Speech Output** — Converts recognized text into audio using Text-to-Speech.
* **Image Preprocessing** — Includes resizing, normalization, and data augmentation to improve model performance.
* **Interactive Interface** — Provides a user-friendly interface for real-time recognition.

## 🛠️ Tech Stack

**Python • TensorFlow • OpenCV • MediaPipe • CNN • NumPy • Streamlit • pyttsx3**

## 🔄 System Workflow

```text
Webcam Input
     ↓
Hand Detection & Landmark Extraction
     ↓
Image Preprocessing
     ↓
CNN-based Gesture Classification
     ↓
Recognized Sign
     ↓
Text Output
     ↓
Speech / Audio Output
```

## 🧠 Model & Dataset

* Custom image dataset collected for different sign-language gestures.
* Images are resized to **64×64** pixels and normalized before training.
* Data augmentation is applied to improve generalization.
* CNN model is trained to classify the predefined gesture categories.
* Achieved approximately **70% classification accuracy** on the project dataset.

## 📊 Applications

* Accessibility and inclusive communication
* Sign-language learning systems
* Assistive communication tools
* Human-computer interaction
* AI-based educational applications

## 🎯 Project Impact

This project demonstrates how **Computer Vision, Deep Learning, and NLP-related speech technologies** can be combined to build an assistive system for real-time sign-language communication.

## ⚠️ Disclaimer

This is an **academic/educational project** developed for demonstration and learning purposes. Recognition performance may vary depending on lighting, camera quality, hand position, and the gesture dataset.
