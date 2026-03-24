# 🧠 Face Recognition System using OpenCV & LBPH

> A full-scale, intelligent face recognition system built for real-world applications using Python, OpenCV, Haar Cascade Classifiers, and the LBPH algorithm. This project mirrors the backbone of modern biometric access and surveillance systems.

---

## 🚀 Overview

This system is more than just face detection — it’s a **production-grade face recognition pipeline**. From data preprocessing, training, and model serialization to real-time inference, the project covers the complete lifecycle of a computer vision-based identity recognition system.

Originally developed as part of a smart surveillance initiative, it efficiently detects and identifies individuals using only grayscale images — making it highly optimized for edge devices and constrained environments.

---

## 🎯 Key Features

- 🧠 **Custom Face Training**  
  Train the model on any dataset of people. Just drop face images into class folders — no annotations required.

- 🖼️ **Real-Time Face Recognition**  
  Detect and recognize faces instantly from static images or live webcam streams.

- 🔍 **LBPH (Local Binary Patterns Histogram)**  
  A powerful algorithm that works well under various lighting conditions and is ideal for grayscale input.

- 🧠 **Haar Cascade-Based Detection**  
  Fast and lightweight facial detection engine built on classic CV techniques.

- 💾 **Model Persistence**  
  Save and load trained face encodings and labels using OpenCV’s built-in methods and NumPy serialization.

- 🧠 **Confidence Scoring**  
  Get a probability-like score for each prediction to evaluate the reliability of recognition results.

---

## 📁 Project Structure
```
Face-Recognition-System/
│
├── face_train.py # Trains the face recognition model
├── face_recognition.py # Predicts face identities from test images
├── haar_face.xml # Haar cascade XML file for face detection
├── faces_trained.yml # Saved trained LBPH model
├── features.npy # Numpy array of extracted face ROIs
├── labels.npy # Corresponding labels for training data
│
├── Faces/
│ ├── train/ # Training data directory
│ └── val/ # Validation/test data
│
├── requirements.txt
└── README.md
```

---

## 🛠️ How It Works

### 🧪 Training the Model
The `face_train.py` script:
1. Reads face images from subfolders in `Faces/train/`.
2. Detects faces using Haar Cascades.
3. Extracts and stores facial ROIs (Regions of Interest).
4. Trains an LBPH model using these face encodings.
5. Saves the model (`faces_trained.yml`) and data arrays (`features.npy`, `labels.npy`).

### 🤖 Predicting with the Model
The `face_recognition.py` script:
1. Loads a test image from `Faces/val/`.
2. Converts it to grayscale and detects faces.
3. Uses the trained model to predict the person's identity.
4. Annotates the image with predicted name and confidence level.

---

## 🎓 Tech Stack

- 🐍 Python 3.x
- 📷 OpenCV (4.x)
- 📊 NumPy
- 💻 LBPH + Haar Classifiers
- 🗃️ File I/O + Model Persistence

---

## 🔧 Setup & Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/Face-Recognition-System.git
   cd Face-Recognition-System
