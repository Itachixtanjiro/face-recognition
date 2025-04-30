# realtime_capture.py
import cv2
import os

collection_dir = "face_data"
os.makedirs(collection_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def collect_faces(name):
    cap = cv2.VideoCapture(0)
    count = 0
    person_dir = os.path.join(collection_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Collecting Faces', frame)
        if cv2.waitKey(1) == 27 or count >= 50:  # ESC or 50 images
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = input("Enter the name of the person: ")
    collect_faces(name)
