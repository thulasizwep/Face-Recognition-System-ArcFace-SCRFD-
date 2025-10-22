import cv2
import os
import time

# Ask for user name
name = input("Enter your name: ").strip()

# Create directory for dataset
dataset_dir = os.path.join("dataset", name)
os.makedirs(dataset_dir, exist_ok=True)
print(f"[INFO] Saving images to: {dataset_dir}")

# Choose camera index
# 0 = laptop cam, 1 = external USB webcam
camera_index = 0  # Change this if needed

# Initialize webcam
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[ERROR] Cannot access camera.")
    exit()

count = 0
total_images = 30
delay = 0.5  # seconds between captures

print("[INFO] Starting capture. Press 'q' to quit early.")

while count < total_images:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Show preview
    cv2.imshow("Capture Dataset", frame)

    # Save image
    img_path = os.path.join(dataset_dir, f"{count+1}.png")
    cv2.imwrite(img_path, frame)
    print(f"[INFO] Saved {img_path}")
    count += 1

    # Wait before next capture
    time.sleep(delay)

    # Allow manual quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"[INFO] Done. {count} images saved to {dataset_dir}")
cap.release()
cv2.destroyAllWindows()
