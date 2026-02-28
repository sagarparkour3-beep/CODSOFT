import cv2
import numpy as np
from deepface import DeepFace
import os

class FaceSystem:
    def __init__(self, known_faces_dir="known_faces"):
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.known_faces_dir = known_faces_dir
        
        # Ensure known_faces directory exists
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)

    def detect_faces(self, image):
        """Detects faces in an image using Haar Cascades and returns bounding boxes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        bboxes = []
        for (x, y, w, h) in faces:
            bboxes.append((x, y, w, h))
        return bboxes

    def recognize_face(self, face_img):
        """Recognizes a face image using DeepFace."""
        try:
            # We use enforce_detection=False because we already detected the face
            results = DeepFace.find(img_path=face_img, 
                                   db_path=self.known_faces_dir, 
                                   enforce_detection=False,
                                   silent=True)
            
            if len(results) > 0 and not results[0].empty:
                # Get the first match
                identity = results[0].iloc[0]['identity']
                name = os.path.basename(identity).split('.')[0]
                return name
            else:
                return "Unknown"
        except Exception as e:
            # In a real app, you'd log this silently or just return Unknown
            # print(f"Error in recognition: {e}")
            return "Unknown"

    def process_frame(self, frame):
        """Processes a single frame for detection and recognition."""
        bboxes = self.detect_faces(frame)
        
        for (x, y, w, h) in bboxes:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Crop face for recognition
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                name = self.recognize_face(face_roi)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        return frame
