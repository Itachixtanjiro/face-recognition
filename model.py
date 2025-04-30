# face_model.py
import cv2
import os
import numpy as np

data_dir = "face_data"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0
    target_size = (200, 200)

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[label_id] = person_name

        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                faces.append(img_resized)
                labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save("face_model.yml")
    print(f"\n✅ Training completed successfully! Total people: {len(label_map)}, Total images: {len(faces)}")
    for label, name in label_map.items():
        print(f" - Label {label}: {name}")
        print("📁 Model saved as 'face_model.yml'")
    return label_map



def recognize_faces(label_map):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label_id, confidence = recognizer.predict(face)

            name = label_map.get(label_id, "Unknown") if confidence < 80 else "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1. Train Model\n2. Recognize Faces")
    choice = input("Enter your choice: ")

    if choice == "1":
        label_map = train_model()
        print("Training completed!")
    elif choice == "2":
        label_map = train_model()  # Load latest
        recognize_faces(label_map)
    else:
        print("Invalid option.")
