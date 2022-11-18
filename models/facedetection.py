import sys
import os
sys.path.append(os.path.join('insightface', 'recognition', 'arcface_mxnet', 'common'))
from face_align import norm_crop
import cv2
import numpy as np


class FaceDetection:
    def __init__(self, model_path, conf_threshold=0.95, size_threshold=900, crop_size=112):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.size_threshold = size_threshold
        self.crop_size = crop_size
        
        self.face_detector = cv2.FaceDetectorYN.create(self.model_path, "", (0, 0))
        
    def detect_face(self, image):
        
        image = np.array(image)
        height, width, _ = image.shape
        
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(image)
        
        # Return results
        cropped_faces = []
        loc_faces = []
        
        if faces is None:
            return cropped_faces, loc_faces
        
        for face in faces:
            
            conf, lm, loc = self.analyze_face(face)
            
            # Remove face with low confidence
            if conf < self.conf_threshold:
                pass
            
            # Remove face with small size
            if loc[2] * loc[3] < self.size_threshold:
                pass
            
            cropped_faces.append(self.align_face(image, lm).astype(np.float32))
            loc_faces.append(loc)
        
        return cropped_faces, loc_faces
            
    def analyze_face(self, face):
        conf = face[-1]
        landmarks = np.reshape(face[4:-1], (5, 2))
        locations = list(map(int, face[:4]))
        return conf, landmarks, locations
     
    def align_face(self, image, landmark):
        aligned_face = norm_crop(
            image,
            landmark=landmark,
            image_size=self.crop_size,
            mode='arc_face'
        )
        
        return aligned_face
        
        
        
        
