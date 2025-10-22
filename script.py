import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import os
import pickle
import argparse
import threading
from queue import Queue
import time

# Initialize the model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
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
                        emb = faces[0].normed_embedding
                        known_embeddings.append(emb)
                        known_labels.append(person)
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': known_embeddings, 'labels': known_labels}, f)
    print("Embeddings generated and saved.")

def recognize_faces(img, known_embeddings, known_labels, threshold=0.6):
    faces = app.get(img)
    for face in faces:
        emb = face.normed_embedding
        min_dist = float('inf')
        label = 'unknown'
        best_index = -1
        for i, k_emb in enumerate(known_embeddings):
            dist = cosine(emb, k_emb)
            if dist < min_dist:
                min_dist = dist
                best_index = i
        if min_dist < threshold:
            label = known_labels[best_index]
            bbox = face.bbox.astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

def capture_camera(camera_url, frame_queue, cam_id):
    """Capture frames from a camera in a separate thread"""
    cap = None
    reconnect_delay = 5  # seconds between reconnection attempts
    
    while True:
        try:
            # Handle different URL types
            if camera_url == "0":
                url = 0  # Webcam
            else:
                url = camera_url
            
            cap = cv2.VideoCapture(url)
            
            # Set buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            if not cap.isOpened():
                print(f"Camera {cam_id} failed to open. Retrying in {reconnect_delay} seconds...")
                time.sleep(reconnect_delay)
                continue
                
            print(f"Camera {cam_id} connected successfully")
            
            while True:
                ret, frame = cap.read()
                if ret:
                    # Resize frame to standard size for consistent display
                    frame = cv2.resize(frame, (640, 480))
                    # Clear queue and put latest frame
                    while not frame_queue.empty():
                        try:
                            frame_queue.get_nowait()
                        except:
                            pass
                    frame_queue.put((cam_id, frame))
                else:
                    print(f"Camera {cam_id} frame read failed. Reconnecting...")
                    break
                    
        except Exception as e:
            print(f"Camera {cam_id} error: {e}")
        
        finally:
            if cap is not None:
                cap.release()
            print(f"Camera {cam_id} disconnected. Reconnecting in {reconnect_delay} seconds...")
            time.sleep(reconnect_delay)

def multi_camera_stream(known_embeddings, known_labels):
    # Define camera URLs - EDIT THESE WITH YOUR ACTUAL IP CAMERA URLs
    camera_urls = [
        "http://192.168.137.93:8080/video",  # IP Camera 1
        0,  # Webcam (use 0 instead of "0")
        "http://192.168.137.93:8080/video",  # IP Camera 2 (same as first for testing)
    ]
    
    # Alternative camera URL formats you can try:
    # "http://192.168.137.93:8080"
    # "http://192.168.137.93:8080/stream"
    # "http://192.168.137.93:8080/videofeed"
    # "rtsp://192.168.137.93:8080/h264_ulaw.sdp"
    
    # Create frame queues for each camera
    frame_queues = [Queue(maxsize=2) for _ in range(len(camera_urls))]
    
    # Start camera capture threads
    threads = []
    for i, url in enumerate(camera_urls):
        thread = threading.Thread(target=capture_camera, args=(url, frame_queues[i], i))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Store latest frames from each camera
    current_frames = [None] * len(camera_urls)
    
    print("Starting multi-camera stream. Press 'q' to quit.")
    
    # Create a black placeholder frame
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Camera Offline", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while True:
        # Get latest frames from all cameras
        for i, queue in enumerate(frame_queues):
            if not queue.empty():
                cam_id, frame = queue.get()
                try:
                    # Apply face recognition
                    processed_frame = recognize_faces(frame.copy(), known_embeddings, known_labels)
                    current_frames[cam_id] = processed_frame
                except Exception as e:
                    print(f"Error processing frame from camera {cam_id}: {e}")
                    current_frames[cam_id] = placeholder
        
        # Use placeholder for cameras that aren't working
        for i in range(len(current_frames)):
            if current_frames[i] is None:
                current_frames[i] = placeholder.copy()
                cv2.putText(current_frames[i], f"Camera {i+1} Offline", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine frames into a single display
        if len(current_frames) >= 2:
            # For 3 cameras: arrange in 2x2 grid (last spot empty) or horizontal layout
            if len(current_frames) == 3:
                # Option 1: Horizontal layout
                combined = np.hstack(current_frames)
                
                # Option 2: 2x2 grid (uncomment below if you prefer grid layout)
                # top_row = np.hstack([current_frames[0], current_frames[1]])
                # bottom_placeholder = np.zeros_like(current_frames[0])
                # cv2.putText(bottom_placeholder, "Empty", (250, 240), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # bottom_row = np.hstack([current_frames[2], bottom_placeholder])
                # combined = np.vstack([top_row, bottom_row])
            else:
                # For other numbers of cameras, use simple hstack
                combined = np.hstack(current_frames[:3])  # Show max 3 cameras
            
            # Add camera labels
            for i, frame in enumerate(current_frames):
                if i < 3:  # Only label first 3 cameras
                    h, w = frame.shape[:2]
                    label_x = i * w + 10
                    cv2.putText(combined, f"Camera {i+1}", (label_x, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Multi-Camera Face Recognition', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Camera Face Recognition")
    parser.add_argument('--generate', action='store_true', help="Generate embeddings from dataset")
    parser.add_argument('--image', type=str, help="Path to test image")
    parser.add_argument('--webcam', action='store_true', help="Use webcam for real-time recognition")
    parser.add_argument('--multicam', action='store_true', help="Use multiple IP cameras")
    args = parser.parse_args()

    if args.generate:
        generate_embeddings('dataset')
    else:
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
        
        if args.multicam:
            multi_camera_stream(known_embeddings, known_labels)