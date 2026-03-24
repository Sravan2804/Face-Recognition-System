import os
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
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
        self.eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
        
        if os.path.exists(self.trained_yml):
            self.face_recognizer.read(self.trained_yml)

    def update_people_list(self):
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        self.people = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]

    def align_face(self, face_roi):
        """Standardizes face tilt based on eye positions."""
        gray_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 5)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda x: x[0])
            l_eye, r_eye = eyes[0], eyes[1]
            l_center = (l_eye[0] + l_eye[2]//2, l_eye[1] + l_eye[3]//2)
            r_center = (r_eye[0] + r_eye[2]//2, r_eye[1] + r_eye[3]//2)
            dy, dx = r_center[1] - l_center[1], r_center[0] - l_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            h, w = face_roi.shape[:2]
            M = cv.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            face_roi = cv.warpAffine(face_roi, M, (w, h), flags=cv.INTER_CUBIC)
        return face_roi

    def process_face(self, face_roi):
        """Pro-pipeline: Alignment -> Grayscale -> CLAHE -> Resize."""
        face_roi = self.align_face(face_roi)
        gray = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
        final = self.clahe.apply(gray)
        return cv.resize(final, (200, 200))

    def get_face_roi(self, img):
        (h, w) = img.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                x, y, x1, y1 = max(0, x), max(0, y), min(w, x1), min(h, y1)
                return img[y:y1, x:x1], (x, y, x1, y1)
        return None, None

    def log_attendance(self, name):
        file_exists = os.path.isfile(self.attendance_file)
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(['Name', 'Date', 'Time'])
            now = datetime.now()
            writer.writerow([name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')])

    def collect_data(self):
        name = simpledialog.askstring("Register", "Enter Name:")
        if not name: return
        path = os.path.join(self.train_dir, name)
        os.makedirs(path, exist_ok=True)
        cap = cv.VideoCapture(0)
        count, last_x = 0, 0
        while count < 40:
            ret, frame = cap.read()
            face, coords = self.get_face_roi(frame)
            if face is not None:
                if abs(coords[0] - last_x) > 10: # Only save if moved
                    count += 1
                    cv.imwrite(os.path.join(path, f"{name}_{count}.jpg"), face)
                    last_x = coords[0]
                cv.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                cv.putText(frame, f"Angle Samples: {count}/40 (Move Head)", (20, 40), 2, 0.7, (0, 255, 0), 2)
            cv.imshow("Registering - Slowly Move Your Head", frame)
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
                if img is not None:
                    features.append(self.process_face(img))
                    labels.append(label)
        if features:
            self.face_recognizer.train(np.array(features), np.array(labels))
            self.face_recognizer.save(self.trained_yml)
            messagebox.showinfo("Success", "Model Updated with Smart Alignment!")

    def run_system(self, mode='camera'):
        if not os.path.exists(self.trained_yml): return messagebox.showerror("Error", "Train model first!")
        if mode == 'camera':
            cap = cv.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                face, coords = self.get_face_roi(frame)
                if face is not None:
                    processed = self.process_face(face)
                    lbl, conf = self.face_recognizer.predict(processed)
                    name = self.people[lbl] if conf < 95 else "Unknown"
                    cv.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                    cv.putText(frame, f"{name} ({int(conf)})", (coords[0], coords[1]-10), 2, 0.7, (0, 255, 0), 2)
                    if name != "Unknown": self.log_attendance(name)
                cv.imshow("AI Recognition (Q to Exit)", frame)
                if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("AI Recognition (Q to Exit)", 0) < 0: break
            cap.release()
            cv.destroyAllWindows()
        else:
            path = filedialog.askopenfilename()
            if path:
                img = cv.imread(path)
                face, coords = self.get_face_roi(img)
                if face is not None:
                    processed = self.process_face(face)
                    lbl, conf = self.face_recognizer.predict(processed)
                    name = self.people[lbl] if conf < 95 else "Unknown"
                    cv.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                    cv.putText(img, name, (coords[0], coords[1]-10), 2, 0.8, (0, 255, 0), 2)
                cv.imshow("Result", img)
                cv.waitKey(0)
                cv.destroyAllWindows()

    def view_attendance(self):
        if not os.path.exists(self.attendance_file): return messagebox.showinfo("Info", "No logs yet.")
        win = tk.Toplevel()
        win.title("Attendance Records")
        tree = ttk.Treeview(win, columns=("Name", "Date", "Time"), show='headings')
        for col in ("Name", "Date", "Time"): tree.heading(col, text=col)
        with open(self.attendance_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader: tree.insert('', 'end', values=row)
        tree.pack(expand=True, fill='both')

# --- GUI ---
if __name__ == "__main__":
    app = FaceRecognitionAI()
    root = tk.Tk()
    root.title("Pro AI Face Recognition System")
    root.geometry("400x520")
    tk.Label(root, text="Face Recognition System", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Button(root, text="1. Smart Register (Multi-Angle)", width=35, height=2, command=app.collect_data).pack(pady=5)
    tk.Button(root, text="2. Train Model (Alignment Active)", width=35, height=2, command=app.train_model).pack(pady=5)
    tk.Button(root, text="3. Live Attendance (Camera)", width=35, height=2, bg="#d4edda", command=lambda: app.run_system('camera')).pack(pady=5)
    tk.Button(root, text="4. Check Single Image File", width=35, height=2, bg="#d1ecf1", command=lambda: app.run_system('file')).pack(pady=5)
    tk.Button(root, text="5. View Attendance Logs", width=35, height=2, bg="#fff3cd", command=app.view_attendance).pack(pady=5)
    tk.Button(root, text="Exit", width=15, command=root.quit).pack(pady=20)
    root.mainloop()