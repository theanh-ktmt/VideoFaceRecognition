import onnxruntime as onnxrt
import numpy as np
import cv2
import torch.nn.functional as F
import torch

class FaceRecognition:
    def __init__(self, model_path, threshold=0.90, input_size=160):
        self.model_path = model_path
        self.onnx_session = onnxrt.InferenceSession(self.model_path)
        self.threshold = threshold
        self.input_size = input_size
        
    def get_embedding(self, face_image):
        face_image = cv2.resize(face_image, (self.input_size, self.input_size))
        face_image = np.transpose(face_image, (2, 0, 1))
        # face_image = np.expand_dims(face_image, axis=0)
        face_image = torch.from_numpy(face_image).unsqueeze(0).float()
        face_image.div_(255).sub_(0.5).div_(0.5)
        face_image = face_image.numpy()
        embedding = self.get_onnx_output(face_image)
        return embedding
    
    def compare_embedding(self, face_embed, known_embeds):
        dist = np.sum(np.square(np.subtract(face_embed, known_embeds)), axis=1)
        if min(dist) > self.threshold:
            return -1
        else:
            idx = np.argmin(dist)
            return idx
        
    def get_onnx_output(self, face):
        onnx_inputs = {
            self.onnx_session.get_inputs()[0].name: face
        }
        onnx_output = self.onnx_session.run(None, onnx_inputs)
        onnx_output = torch.tensor(np.array(onnx_output))
        onnx_output = F.normalize(torch.squeeze(onnx_output, 0))
        return onnx_output[0].numpy()