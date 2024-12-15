import cv2
import os


cap = cv2.VideoCapture(0)


dataset_size = 500


data_dir = "Data_Collection"
os.makedirs(data_dir, exist_ok=True)

person_name = input("Enter your name: ").strip()
person_dir = os.path.join(data_dir, person_name)
os.makedirs(person_dir, exist_ok=True)


count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    else:
        cv2.putText(
            frame,
            "Press 'C' to start collection or 'Q' to quit",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Collection", cv2.resize(frame, (800, 700)))

        key = cv2.waitKey(1)

        if key == ord("c"):
            while count < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    break
                img_path = os.path.join(person_dir, f"{person_name}_{count}.png")
                cv2.imwrite(img_path, frame)
                count += 1
                cv2.imshow("Collection", cv2.resize(frame, (800, 700)))
                if cv2.waitKey(1) == ord("q"):
                    break

            break
        if key == ord("q"):
            print("Collection stopped by user.")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
