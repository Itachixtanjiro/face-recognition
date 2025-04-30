import cv2
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance

# Setup
SOURCE_DIR = Path("face_data")
DEST_DIR = Path("preprocessed_data") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Utility Functions
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    return blur

def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def adjust_brightness(image, factor=1.3):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhanced = ImageEnhance.Brightness(pil_img).enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

def process_and_augment(img_path, output_folder):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Skipped unreadable image: {img_path}")
        return

    pre = preprocess_image(img)
    fname = img_path.stem

    cv2.imwrite(str(output_folder / f"{fname}_pre.jpg"), pre)
    cv2.imwrite(str(output_folder / f"{fname}_flip.jpg"), cv2.flip(pre, 1))
    cv2.imwrite(str(output_folder / f"{fname}_rot.jpg"), rotate_image(pre, 15))
    cv2.imwrite(str(output_folder / f"{fname}_bright.jpg"), preprocess_image(adjust_brightness(img)))

# Main logic
img_exts = ['*.jpg', '*.jpeg', '*.png']
img_files = []
for ext in img_exts:
    img_files.extend(SOURCE_DIR.rglob(ext))

if not img_files:
    print("[ERROR] No image files found in subfolders of face_data/. Check extensions or structure.")
else:
    for img_path in img_files:
        rel_path = img_path.relative_to(SOURCE_DIR).parent
        output_folder = DEST_DIR / rel_path
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Processing image: {img_path}")
        process_and_augment(img_path, output_folder)

    print(f"\n✅ Preprocessing complete. Output saved to: {DEST_DIR}")


