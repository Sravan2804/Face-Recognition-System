import os
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from datetime import datetime
import csv

class FaceRecognitionAI:
    def __init__(self):
        # 1. Path Configuration
        self.base_path = r'D:\Projects\OpenCV\Face-Recognition-System'
        self.proto_path = os.path.join(self.base_path, 'deploy.prototxt.txt')
        self.model_path = os.path.join(self.base_path, 'res10_300x300_ssd_iter_140000.caffemodel')
        self.train_dir = os.path.join(self.base_path, r'Faces\train')
        self.trained_yml = os.path.join(self.base_path, 'faces_trained.yml')
        self.attendance_file = os.path.join(self.base_path, 'attendance.csv')
        
        self.update_people_list()
        
        # 2. Model Initialization
        self.net = cv.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        self.face_recognizer = cv.face.LBPHFaceRecognizer_create()
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Eye detector for alignment
        self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
        
        if os.path.exists(self.trained_yml):
            self.face_recognizer.read(self.trained_yml)

    def update_people_list(self):
        if os.path.exists(self.train_dir):
            self.people = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        else:
            self.people = []

    def align_face(self, gray_img, face_roi):
        """Rotates the face so eyes are horizontal (Alignment)."""
        eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate to find left and right
            eyes = sorted(eyes, key=lambda x: x[0])
            left_eye_center = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            right_eye_center = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)

            # Calculate angle between eye centers
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dY, dX))

            # Rotate image around the center of the face
            h, w = face_roi.shape[:2]
            center = (w // 2, h // 2)
            M = cv.getRotationMatrix2D(center, angle, 1.0)
            face_roi = cv.warpAffine(face_roi, M, (w, h), flags=cv.INTER_CUBIC)
            
        return face_roi

    def process_face(self, face_roi):
        """Grayscale -> Alignment -> CLAHE -> Resize."""
        gray = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
        aligned = self.align_face(gray, gray)
        final = self.clahe.apply(aligned)
        return cv.resize(final, (200, 200))

    def get_face_roi(self, img):
        (h, w) = img.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                x, y, x1, y1 = max(0, x), max(0, y), min(w, x1), min(h, y1)
                face_roi = img[y:y1, x:x1]
                if face_roi.size == 0: continue
                return face_roi, (x, y, x1, y1)
        return None, None

    def log_attendance(self, name):
        """Saves recognized person to a CSV file with timestamp."""
        file_exists = os.path.isfile(self.attendance_file)
        with open(self.attendance_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Name', 'Date', 'Time'])
            
            now = datetime.now()
            writer.writerow([name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])

    def collect_data(self):
        name = simpledialog.askstring("Register", "Enter Name:")
        if not name: return
        path = os.path.join(self.train_dir, name)
        os.makedirs(path, exist_ok=True)
        
        cap = cv.VideoCapture(0)
        count, ts = 0, int(datetime.now().timestamp())
        while count < 30:
            ret, frame = cap.read()
            face, coords = self.get_face_roi(frame)
            if face is not None:
                count += 1
                cv.imwrite(os.path.join(path, f"{ts}_{count}.jpg"), face)
                x, y, x1, y1 = coords
                cv.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
            cv.imshow("Registering - Stay Still & Tilt Head Slightly", frame)
            if cv.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv.destroyAllWindows()
        self.update_people_list()

    def train_model(self):
        self.update_people_list()
        features, labels = [], []
        for person in self.people:
            path = os.path.join(self.train_dir, person)
            label = self.people.index(person)
            for img_name in os.listdir(path):
                img = cv.imread(os.path.join(path, img_name))
                if img is None: continue
                features.append(self.process_face(img))
                labels.append(label)
        
        if features:
            self.face_recognizer.train(np.array(features), np.array(labels))
            self.face_recognizer.save(self.trained_yml)
            messagebox.showinfo("Success", "Model trained with Face Alignment!")

    def recognize_logic(self, frame):
        face, coords = self.get_face_roi(frame)
        if face is not None:
            processed = self.process_face(face)
            label, conf = self.face_recognizer.predict(processed)
            name = self.people[label] if conf < 90 else "Unknown"
            
            # Draw
            x, y, x1, y1 = coords
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv.rectangle(frame, (x, y), (x1, y1), color, 2)
            cv.putText(frame, f"{name} ({int(conf)})", (x, y-10), 2, 0.7, color, 2)
            
            if name != "Unknown":
                self.log_attendance(name)
        return frame

    def run_camera(self):
        cap = cv.VideoCapture(0)
        win = "Pro AI Recognition (Q to Exit)"
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv.imshow(win, self.recognize_logic(frame))
            if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv.destroyAllWindows()

    def run_file(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return
        img = cv.imread(file_path)
        result = self.recognize_logic(img)
        cv.imshow('File Result', result)
        cv.waitKey(0)
        cv.destroyAllWindows()

# --- MAIN GUI ---
if __name__ == "__main__":
    app = FaceRecognitionAI()
    root = tk.Tk()
    root.title("Pro Face AI Suite 2026")
    root.geometry("400x500")
    
    tk.Label(root, text="Pro Face Recognition AI", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Button(root, text="1. Register Person (Alignment Active)", width=35, height=2, command=app.collect_data).pack(pady=5)
    tk.Button(root, text="2. Train Model (CLAHE Optimized)", width=35, height=2, command=app.train_model).pack(pady=5)
    
    tk.Label(root, text="System Output:", font=("Arial", 10, "italic")).pack(pady=10)
    tk.Button(root, text="Start Live Attendance", width=35, height=2, bg="#d4edda", command=app.run_camera).pack(pady=5)
    tk.Button(root, text="Test Single Image File", width=35, height=2, bg="#d1ecf1", command=app.run_file).pack(pady=5)
    
    tk.Button(root, text="Exit", width=15, command=root.quit, bg="#f8d7da").pack(pady=30)
    root.mainloop()