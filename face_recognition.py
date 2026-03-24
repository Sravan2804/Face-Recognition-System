# import os
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# from mediapipe.python.solutions import face_mesh as mp_face_mesh
# from mediapipe.python.solutions import drawing_utils as mp_drawing

# har_cascade = cv.CascadeClassifier('haar_face.xml')

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# def get_aligned_face(img):
#     h, w = img.shape[:2]
#     rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_img)
    
#     if not results.multi_face_landmarks:
#         return None, None

#     landmarks = results.multi_face_landmarks[0].landmark
#     # Landmarks for eyes
#     left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
#     right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])

#     # Calculate angle and rotate
#     dY, dX = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
#     angle = np.degrees(np.arctan2(dY, dX))
#     eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    
#     M = cv.getRotationMatrix2D(eye_center, angle, 1.0)
#     rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC)
    
#     # Simple crop logic based on eye center
#     # This ensures the face is centered for the LBPH recognizer
#     gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
#     size = 200
#     x_start, y_start = max(0, eye_center[0]-100), max(0, eye_center[1]-100)
#     face_roi = gray[y_start:y_start+200, x_start:x_start+200]
    
#     if face_roi.shape[:2] != (200, 200):
#         face_roi = cv.resize(face_roi, (200, 200))
        
#     return face_roi, (x_start, y_start, 200, 200)

# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Sravan']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy', allow_pickle=True)

# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read('faces_trained.yml')

# img = cv.imread(r'D:\Projects\OpenCV\Face-Recognition-System\Faces\val\ben_afflek\5.jpg')
# aligned_face, coords = get_aligned_face(img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', img)

# #faces_rect = har_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# if aligned_face is not None:
#     label, confidence = face_recognizer.predict(aligned_face)
#     print(f'Detected: {people[label]} with confidence {confidence}')

#     # Draw on original image
#     x, y, w, h = coords
#     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     cv.putText(img, f'{people[label]} ({int(confidence)})', (x, y-10), 
#                cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

# cv.imshow('Improved Recognition', img)
# cv.waitKey(0)




# # for (x, y, w, h) in faces_rect:
# #     faces_roi = gray[y:y+h, x:x+w]
# #     label, confidence = face_recognizer.predict(faces_roi)
    
# #     print(f'Label: {people[label]}, Confidence: {confidence}')

# #     cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness = 2)
# #     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

# cv.imshow('Detected Face', img)
# cv.waitKey(0)

import cv2 as cv
import numpy as np

# Load the DNN Face Detector
proto_path = r'D:\Projects\OpenCV\Face-Recognition-System\deploy.prototxt.txt'
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv.dnn.readNetFromCaffe(proto_path, model_path)

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Sravan']

# Load the trained LBPH model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faces_trained.yml')

# Load the image to recognize
img = cv.imread(r'D:\Projects\OpenCV\Face-Recognition-System\Faces\val\ben_afflek\5.jpg')
(h, w) = img.shape[:2]

# Detect face using DNN
blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence_det = detections[0, 0, i, 2]
    
    if confidence_det > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")
        
        # Crop and process for recognizer
        face_roi = img[y:y1, x:x1]
        if face_roi.size == 0: continue
        
        gray_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
        gray_roi = cv.resize(gray_roi, (200, 200))
        
        # Predict the person
        label, confidence_rec = face_recognizer.predict(gray_roi)
        
        # LBPH Confidence: Lower is better (closer to 0 is a perfect match)
        result_text = f'{people[label]} ({int(confidence_rec)})'
        
        # Drawing
        cv.rectangle(img, (x, y), (x1, y1), (0, 255, 0), thickness=2)
        cv.putText(img, result_text, (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        
        print(f'Detected: {people[label]} | Confidence: {confidence_rec}')

cv.imshow('Face Recognition System', img)
cv.waitKey(0)