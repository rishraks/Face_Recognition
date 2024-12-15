import cv2
import mediapipe as mp
import pickle
import numpy as np
import os


cap = cv2.VideoCapture(0)

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

mp_detect_face = mp.solutions.face_detection

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_iris = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
mp_drawing_contours = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

labels_dict = {folder: folder for folder in os.listdir("./Data_Collection")}


with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
) as mesh:
    with mp_detect_face.FaceDetection(
        model_selection=0, min_detection_confidence=0.6
    ) as detect:
        while True:
            data_aux = []
            ret, frame = cap.read()
            if not ret:
                break
            else:
                # Face Detection
                H, W, _ = frame.shape
                output = detect.process(frame)
                if output.detections is not None:
                    for detections in output.detections:
                        location_data = detections.location_data
                        bbox = location_data.relative_bounding_box
                        x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                        x1 = int(x1 * W)
                        y1 = int(y1 * H)
                        w = int(w * W)
                        h = int(h * H)
                        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

                # Face Meshing
                results = mesh.process(frame)
                if results.multi_face_landmarks is not None:
                    # for landmarks in results.multi_face_landmarks:
                    #     mp_drawing.draw_landmarks(
                    #         image=frame,
                    #         landmark_list=landmarks,
                    #         connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #         landmark_drawing_spec=None,
                    #         connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    #     )
                    #     mp_drawing.draw_landmarks(
                    #         image=frame,
                    #         landmark_list=landmarks,
                    #         connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    #         landmark_drawing_spec=None,
                    #         connection_drawing_spec=mp_drawing_contours,
                    #     )

                    #     mp_drawing.draw_landmarks(
                    #         image=frame,
                    #         landmark_list=landmarks,
                    #         connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    #         landmark_drawing_spec=None,
                    #         connection_drawing_spec=mp_drawing_iris,
                    #     )

                    for landmarks in results.multi_face_landmarks:
                        for i in range(len(landmarks.landmark)):
                            x = landmarks.landmark[i].x
                            y = landmarks.landmark[i].y
                            data_aux.append(x)
                            data_aux.append(y)

                    prediction = model.predict([np.asarray(data_aux)])

                    predicted_person = labels_dict[(prediction[0])]
                    cv2.putText(
                        frame,
                        predicted_person,
                        (x1-4, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("Testing", frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

cap.release()
cv2.destroyAllWindows()
