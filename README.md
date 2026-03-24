# 🧠 Face Recognition System using OpenCV & LBPH

> An end-to-end Face Recognition Suite built with OpenCV DNN and LBPH (Local Binary Patterns Histograms). This system goes beyond basic recognition by implementing "Smart Registration" (capturing faces from multiple angles), Eye-based Face Alignment, and CLAHE lighting normalization.
---

## 🚀 Overview

This system is more than just face detection — it’s a **production-grade face recognition pipeline**. From data preprocessing, training, and model serialization to real-time inference, the project covers the complete lifecycle of a computer vision-based identity recognition system.

Originally developed as part of a smart surveillance initiative, it efficiently detects and identifies individuals using only grayscale images — making it highly optimized for edge devices and constrained environments.

---

## 🎯 Key Features

- 🧠 **DNN Face Detection**  
  Uses a ResNet-10 SSD Caffe model for high-accuracy face detection (far superior to traditional Haar Cascades).

- 🖼️ **Smart Registration (FaceID Style):**  
  A guided data collection process that ensures the model learns your face from multiple angles (Center, Left, Right, Up, Down).

- 🔍 **LBPH (Local Binary Patterns Histogram)**  
  A powerful algorithm that works well under various lighting conditions and is ideal for grayscale input.

- 🧠 **Face Alignment**  
  Automatically detects eyes and rotates the face to a neutral, horizontal position to increase recognition accuracy.

- 💾 **Lighting Normalization (CLAHE)**  
  Implements Contrast Limited Adaptive Histogram Equalization to handle harsh shadows and inconsistent lighting.

- 🧠 **Built-in Attendance System**  
  Automatically logs recognized individuals into a attendance.csv file with precise timestamps.

- 🧠 **Interactive GUI**  
  A clean Tkinter interface for registering users, training the model, and running live recognition.

---

## 📁 Project Structure
```
Face-Recognition-System/
│
├── deploy.prototxt.txt      # DNN architecture file
├── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained weights
├── faces_trained.yml        # The trained LBPH model (generated)
├── attendance.csv           # Attendance logs (generated)
└── main.py                  # The complete system code
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

1. Detection: Frame is converted to a blob and passed through the SSD Caffe Model.
2. Alignment: Eye centers are located; the face is rotated to a 0 degree tilt.
3. Preprocessing: Image is converted to grayscale and CLAHE is applied to normalize histograms.
4. Classification: LBPH predicts the ID and provides a confidence score (distance).

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

2. **Install Dependencies**
  ```bash
    pip install opencv-contrib-python numpy
    ```

3. **Download Model Files**
  Ensure deploy.prototxt.txt and res10_300x300_ssd_iter_140000.caffemodel are in the root directory.


## Usage

Run the application:

```bash
  python main.py
  ```
1. Register a Person: Click Step 1, enter a name, and slowly move your head as instructed by the on-screen prompts.

2. Train the AI: Click Step 2. The system will align all captured images and build the faces_trained.yml file.

3. Start Recognition: Choose Live Attendance for webcam use or Upload Image to test a specific file.

4. View Logs: Click View Attendance Logs to see a history of recognized users directly within the app.