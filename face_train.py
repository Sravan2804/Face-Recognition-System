import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Sravan']
DIR = r'D:\Projects\OpenCV\Face-Recognition-System\Faces\train'

features = []
labels = []
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

def prepare_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

prepare_train()

print(f'Training is done with {len(features)} features and {len(labels)} labels')

features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#train the recognizer
face_recognizer.train(features, labels)
# save the trained model
face_recognizer.save('faces_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)



