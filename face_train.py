# import os
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# from mediapipe.python.solutions import face_mesh as mp_face_mesh
# from mediapipe.python.solutions import drawing_utils as mp_drawing

# # Initialize MediaPipe Face Mesh (468 points)
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Sravan']
# DIR = r'D:\Projects\OpenCV\Face-Recognition-System\Faces\train'

# features = []
# labels = []
# # haar_cascade = cv.CascadeClassifier('haar_face.xml')

# # faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# def get_aligned_face(img):
#     """Detects, aligns, and crops the face."""
#     h, w = img.shape[:2]
#     rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_img)
    
#     if not results.multi_face_landmarks:
#         return None

#     # Get landmarks for eyes (Indices: Left Eye 33, Right Eye 263)
#     landmarks = results.multi_face_landmarks[0].landmark
#     left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
#     right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])

#     # Calculate angle to make eyes horizontal
#     dY = right_eye[1] - left_eye[1]
#     dX = right_eye[0] - left_eye[0]
#     angle = np.degrees(np.arctan2(dY, dX))

#     # Rotate image around the center of the eyes
#     eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
#     M = cv.getRotationMatrix2D(eye_center, angle, 1.0)
#     rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC)

#     # Convert to gray and crop the face area (simplified crop for this example)
#     gray = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
#     return cv.resize(gray, (200, 200)) # Standardize size

# def prepare_train():
#     for person in people:
#         path = os.path.join(DIR, person)
#         label = people.index(person)

#         for img_name in os.listdir(path):
#             img_path = os.path.join(path, img_name)
#             img_array = cv.imread(img_path)
#             if img_array is None: continue

#             # USE ALIGNMENT INSTEAD OF HAAR
#             face_roi, _ = get_aligned_face(img_array)
            
#             if face_roi is not None:
#                 features.append(face_roi)
#                 labels.append(label)

# prepare_train()

# print(f'Training is done with {len(features)} features and {len(labels)} labels')

# features = np.array(features, dtype = 'object')
# labels = np.array(labels)

# face_recognizer = cv.face.LBPHFaceRecognizer_create()

# #train the recognizer
# face_recognizer.train(features, labels)
# # save the trained model
# face_recognizer.save('faces_trained.yml')
# np.save('features.npy', features)
# np.save('labels.npy', labels)



import os
import cv2 as cv
import numpy as np

# Load the DNN Face Detector
proto_path = r'D:\Projects\OpenCV\Face-Recognition-System\deploy.prototxt.txt'
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv.dnn.readNetFromCaffe(proto_path, model_path)

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Sravan']
DIR = r'D:\Projects\OpenCV\Face-Recognition-System\Faces\train'

features = []
labels = []

def get_face_roi(img):
    """Detects and returns a standardized 200x200 face."""
    (h, w) = img.shape[:2]
    # Blob preprocessing for the DNN
    blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Filter weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            
            # Ensure coordinates are within image boundaries
            x, y = max(0, x), max(0, y)
            x1, y1 = min(w, x1), min(h, y1)
            
            face_roi = img[y:y1, x:x1]
            if face_roi.size == 0: continue
            
            gray_face = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
            return cv.resize(gray_face, (200, 200))
    return None

def prepare_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_array = cv.imread(img_path)
            if img_array is None: continue

            face = get_face_roi(img_array)
            if face is not None:
                features.append(face)
                labels.append(label)

print("Training started...")
prepare_train()
print(f'Training complete: {len(features)} faces processed.')

# Create and train LBPH Recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(np.array(features), np.array(labels))

# Save the trained model and data
face_recognizer.save('faces_trained.yml')
np.save('features.npy', np.array(features, dtype='object'))
np.save('labels.npy', np.array(labels))
print("Model saved as faces_trained.yml")