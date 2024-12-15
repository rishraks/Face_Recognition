import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import pickle


data_dir = "Data_Collection"

mp_face_mesh = mp.solutions.face_mesh


data = []
labels = []

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
) as mesh:
    for dir_ in os.listdir(data_dir):
        for img_path in os.listdir(os.path.join(data_dir, dir_)):
            data_aux = []
            img = cv2.imread(os.path.join(data_dir, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = mesh.process(img_rgb)
            if results.multi_face_landmarks is not None:
                for landmarks in results.multi_face_landmarks:
                    for i in range(len(landmarks.landmark)):
                        x = landmarks.landmark[i].x
                        y = landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)

f = open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()