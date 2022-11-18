import sys
sys.path.append(r'D:\Workplaces\Python\BKC\VideoFaceRecognition')
import yaml
import os
from pathlib import Path
from PIL import Image
from models import FrameProcessing
import pickle
import numpy as np

# Load config
with open('configs\\main_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Init model
process = FrameProcessing(config)
detector = process.face_detector
recognizer = process.face_recognizer

# Embed all face from a image
def get_id(image):
    # Detect face from image
    cropped_faces, _ = detector.detect_face(image)
    if len(cropped_faces) > 0:
        cropped_face = cropped_faces[0]
        # Get face embedding
        face_emb = recognizer.get_embedding(cropped_face)
        return face_emb
    else:
        return None

# Load 
if os.path.exists(config['known_id_path']) and not config['reset_ids']:
    known_ids = pickle.load(open(config['known_id_path'], 'rb'))
else:
    # Create dir if not exist
    Path(os.path.sep.join(config['known_id_path'].split(os.path.sep)[:-1])).mkdir(exist_ok=True, parents=True)
    
    # Reset or create new id list
    known_ids = {}

known_people = os.listdir(config['known_image_folder'])
print('Processing {} faces in dataset ...'.format(len(known_people)))
for person in known_people:
    msg = 'Added'
    if person not in known_ids:
        # Prepare path
        src_path = os.path.join(config['known_image_folder'], person)
        
        # Get all face embeddings from face dataset
        embs = []
        for image in os.listdir(src_path):
            path = os.path.join(src_path, image)
            image = Image.open(path)
            emb = get_id(image)
            if emb is not None:
                embs.append(emb)
                
        if len(embs) > 0:
            # Get mean embedding
            mean_emb = np.mean(np.array(embs), axis=0)
            known_ids[person] = mean_emb
        else:
            msg = 'Bad image'
    else:
        msg = 'Existed'
        
    print('- {}: {}'.format(person, msg))
    
print()

# Save ids file
pickle.dump(known_ids, open(config['known_id_path'], 'wb'))
print('Saved known ids to {}!'.format(config['known_id_path']))
        
