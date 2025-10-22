import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import os
import pickle
import argparse

# Initialize the model (uses ArcFace for recognition, SCRFD for detection - better than MTCNN for accuracy)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) # Use 'CUDAExecutionProvider' if you have GPU
app.prepare(ctx_id=0, det_size=(640, 640))

def generate_embeddings(dataset_dir):
    known_embeddings = []
    known_labels = []
    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    img = cv2.imread(img_path)
                    faces = app.get(img)
                    if len(faces) > 0:
                        # Assume one face per dataset image; take the first
                        emb = faces[0].normed_embedding
                        known_embeddings.append(emb)
                        known_labels.append(person)
    # Save embeddings and labels
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': known_embeddings, 'labels': known_labels}, f)
    print("Embeddings generated and saved.")

def recognize_faces(img, known_embeddings, known_labels, threshold=0.6):  # Threshold for cosine distance (lower = stricter)
    faces = app.get(img)
    for face in faces:
        emb = face.normed_embedding
        min_dist = float('inf')
        label = 'unknown'
        best_index = -1
        for i, k_emb in enumerate(known_embeddings):
            dist = cosine(emb, k_emb)  # Cosine distance (0 = identical, 2 = opposite)
            if dist < min_dist:
                min_dist = dist
                best_index = i
        if min_dist < threshold:
            label = known_labels[best_index]
            # Only draw box and label for recognized people
            bbox = face.bbox.astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Skip drawing for unknown faces
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition with MTCNN-like Detection and ArcFace")
    parser.add_argument('--generate', action='store_true', help="Generate embeddings from dataset")
    parser.add_argument('--image', type=str, help="Path to test image")
    parser.add_argument('--webcam', action='store_true', help="Use webcam for real-time recognition")
    args = parser.parse_args()

    if args.generate:
        generate_embeddings('dataset')
    else:
        # Load embeddings
        if not os.path.exists('embeddings.pkl'):
            print("Run with --generate first to create embeddings.pkl")
            exit()
        with open('embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        known_embeddings = data['embeddings']
        known_labels = data['labels']

        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print("Invalid image path")
                exit()
            result = recognize_faces(img, known_embeddings, known_labels)
            output_path = 'output_' + os.path.basename(args.image)
            cv2.imwrite(output_path, result)
            print(f"Processed image saved as {output_path}")

        if args.webcam:
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = recognize_faces(frame, known_embeddings, known_labels)
                cv2.imshow('Face Recognition', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()