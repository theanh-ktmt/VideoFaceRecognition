from .facedetection import FaceDetection
from .facerecognition import FaceRecognition
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import os
import uuid

class FrameProcessing:
    def __init__(self, config):
        self.config = config
        
        # Init face detection
        self.face_detector = FaceDetection(
            config['face_detection_path'], 
            crop_size=self.config['detect_size'],
            conf_threshold=self.config['detect_conf_threshold'],
            size_threshold=self.config['detect_size_threshold']  # px
        )
        
        # Init face recogntion
        self.face_recognizer = FaceRecognition(
            self.config['face_recognition_path'],
            threshold=self.config['recognize_threshold'],
            input_size=self.config['recognize_input_size']
        )
    
    # Full process of face processing
    def process_frame(self, frame, known_ids):
        # Detect face with face detector
        cropped_faces, _ = self.detect_face(frame)
        
        # Iter all faces in that frame
        recognized_people = []
        for face in cropped_faces:
            
            # Get face embedding
            face_emb = self.get_embedding(face)
            
            # Get face index
            person_index = self.compare_embedding(face_emb, list(known_ids.values()))
            
            # Get face name
            if person_index != -1:
                name = list(known_ids.keys())[person_index]
                recognized_people.append(name)
                if self.config['log_results']:
                    self.log(name, face.astype(np.uint8)) 
            
        return recognized_people
            
    # Log results to dir
    def log(self, name, face):
        name_dir = os.path.join(self.config['log_dir'], name)
        Path(name_dir).mkdir(exist_ok=True, parents=True)
        path = os.path.join(self.config['log_dir'], name, '{}.jpg'.format(uuid.uuid4()))
        plt.imsave(path, face)
    
    # Detect face from image
    def detect_face(self, image):
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.face_detector.detect_face(image)
        else:
            return [], []
    
    # Get embedding of a face
    def get_embedding(self, face):
        return self.face_recognizer.get_embedding(face)
    
    # Compare that face embedding with known embedding
    def compare_embedding(self, face_embed, known_embeds):
        return self.face_recognizer.compare_embedding(face_embed, known_embeds)
    
        