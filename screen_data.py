import os
import cv2
import numpy as np
import time
from datetime import datetime
import pyautogui
from PIL import Image, ImageEnhance
import uuid

# Set global configurations
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
output_root = "face_data"
face_difference_threshold = 5000
frame_interval = 1.0  # seconds

# Ask user for duration input (limit to 1 hour)
while True:
    try:
        duration_minutes = float(input("Enter capture duration in minutes (max 60): "))
        if 0 < duration_minutes <= 60:
            break
        else:
            print("⚠️ Please enter a value between 1 and 60.")
    except ValueError:
        print("❌ Invalid input. Enter a numeric value.")

end_time = time.time() + duration_minutes * 60
session_folder = os.path.join(output_root, datetime.now().strftime("session_%Y%m%d_%H%M%S"))
os.makedirs(session_folder, exist_ok=True)

# Initialize
last_saved_time = 0
last_face = None

print(f"[🔍] Capturing started. Data will be saved in: {session_folder}")

while time.time() < end_time:
    try:
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Resize to fixed size for consistency
            face_img = cv2.resize(face_img, (100, 100))

            # Skip if same as last captured
            if last_face is not None:
                diff = np.sum(cv2.absdiff(face_img, last_face))
                if diff < face_difference_threshold:
                    continue

            # Enhance image (augmentation)
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
            pil_img = ImageEnhance.Brightness(pil_img).enhance(1.2)
            face_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Save the face image
            filename = os.path.join(session_folder, f"face_{uuid.uuid4().hex}.png")
            cv2.imwrite(filename, face_img)
            print(f"[📸] Saved: {filename}")

            last_saved_time = time.time()
            last_face = face_img

        time.sleep(frame_interval)

    except KeyboardInterrupt:
        print("⛔ Capture interrupted by user.")
        break

print("✅ Face capture session complete.")



