import os
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

class FaceRecognitionSystem:
    def __init__(self):
        self.people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Sravan']
        self.base_path = r'D:\Projects\OpenCV\Face-Recognition-System'
        self.proto_path = os.path.join(self.base_path, 'deploy.prototxt.txt')
        self.model_path = os.path.join(self.base_path, "res10_300x300_ssd_iter_140000.caffemodel")
        self.train_dir = os.path.join(self.base_path, r'Faces\train')
        
        # Load DNN Detector
        self.net = cv.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        
        # Initialize Recognizer
        self.face_recognizer = cv.face.LBPHFaceRecognizer_create()
        if os.path.exists('faces_trained.yml'):
            self.face_recognizer.read('faces_trained.yml')

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

    def collect_data(self):
        name = simpledialog.askstring("Input", "Enter the name of the person:")
        if not name: return
        
        path = os.path.join(self.train_dir, name)
        os.makedirs(path, exist_ok=True)
        if name not in self.people: self.people.append(name)
        
        cap = cv.VideoCapture(0)
        count = 0
        while count < 30:
            ret, frame = cap.read()
            face, coords = self.get_face_roi(frame)
            if face is not None:
                count += 1
                cv.imwrite(os.path.join(path, f"{count}.jpg"), face)
                x, y, x1, y1 = coords
                cv.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
                cv.putText(frame, f"Saved: {count}/30", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv.imshow("Collecting Data - Press Q to Cancel", frame)
            if cv.waitKey(1) & 0xFF == ord('q'): break
        
        cap.release()
        cv.destroyAllWindows()
        messagebox.showinfo("Success", f"Collected 30 samples for {name}")

    def train_model(self):
        features, labels = [], []
        for person in self.people:
            path = os.path.join(self.train_dir, person)
            if not os.path.exists(path): continue
            label = self.people.index(person)
            for img_name in os.listdir(path):
                img_array = cv.imread(os.path.join(path, img_name))
                if img_array is None: continue
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
                features.append(cv.resize(gray, (200, 200)))
                labels.append(label)
        
        self.face_recognizer.train(np.array(features), np.array(labels))
        self.face_recognizer.save('faces_trained.yml')
        messagebox.showinfo("Training", "Model trained and saved successfully!")

    def run_camera(self):
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            face, coords = self.get_face_roi(frame)
            if face is not None:
                gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                label, conf = self.face_recognizer.predict(cv.resize(gray, (200, 200)))
                name = self.people[label] if conf < 100 else "Unknown"
                x, y, x1, y1 = coords
                cv.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                cv.putText(frame, f"{name} ({int(conf)})", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv.imshow('Press Q to Exit Camera', frame)
            if cv.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv.destroyAllWindows()

    def run_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return
        img = cv.imread(file_path)
        face, coords = self.get_face_roi(img)
        if face is not None:
            gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            label, conf = self.face_recognizer.predict(cv.resize(gray, (200, 200)))
            name = self.people[label] if conf < 100 else "Unknown"
            x, y, x1, y1 = coords
            cv.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
            cv.putText(img, f"{name} ({int(conf)})", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.imshow('Result', img)
        cv.waitKey(0)

# --- GUI SETUP ---
root = tk.Tk()
root.title("Face Recognition AI")
root.geometry("300x400")

sys = FaceRecognitionSystem()

tk.Label(root, text="Face AI Control Panel", font=("Arial", 14, "bold")).pack(pady=20)
tk.Button(root, text="1. Collect Face Data", width=25, command=sys.collect_data).pack(pady=5)
tk.Button(root, text="2. Train Model", width=25, command=sys.train_model).pack(pady=5)
tk.Button(root, text="3. Live Camera Recognition", width=25, bg="lightgreen", command=sys.run_camera).pack(pady=5)
tk.Button(root, text="4. Recognize from Image", width=25, bg="lightblue", command=sys.run_image).pack(pady=5)
tk.Button(root, text="Exit", width=25, command=root.quit).pack(pady=20)

root.mainloop()