from models import VideoProcessing

import yaml
import os
import pickle
import pafy
import streamlit as st
from pathlib import Path
from PIL import Image
import datetime

# Load configs
with open(os.path.join('configs', 'main_config.yaml')) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Init video process
videoprocess = VideoProcessing(config)

# Get known ids
known_ids = pickle.load(open(config['known_id_path'], 'rb'))

# Sidebar config
st.sidebar.title('Options')
st.sidebar.markdown('---')

st.sidebar.markdown('**Face detection**')
config['detect_size'] = st.sidebar.select_slider('Detect size', options=[112, 224, 448], value=config['detect_size'])
config['detect_conf_threshold'] = st.sidebar.number_input('Confident threshold', min_value=0., max_value=2., step=0.05, value=config['detect_conf_threshold'])
config['detect_size_threshold'] = st.sidebar.number_input('Size threshold', min_value=100, max_value=10000, step=1, value=config['detect_size_threshold'])

st.sidebar.markdown('---')

st.sidebar.markdown('**Face recognition**')
pretrained = st.sidebar.selectbox('Pretrained model', options=['Pixta', 'VGGFace2'], index=0)
if pretrained == 'Pixta':
    config['recognize_threshold'] = st.sidebar.number_input('Confident threshold', min_value=0., max_value=2., step=0.05, value=0.95)
    config['recognize_input_size'] = 112
elif pretrained == 'VGGFace2':
    config['recognize_threshold'] = st.sidebar.number_input('Confident threshold', min_value=0., max_value=2., step=0.05, value=0.05)
    config['recognize_input_size'] = 160

# Main screen config
st.title('BVideo')
st.subheader('Video face recognition')
st.markdown('---')

st.header('Youtube')
config['video_path'] = st.text_input('Link', value='')
thumb = st.image([])
st.markdown('---')

st.header('Conclusion')

if config['video_path'] != '':

    st.markdown('**Progress**')
    progress = st.progress(0)
    
    # Get person name in video
    if 'youtube' in config['video_path']:
        video = pafy.new(config['video_path'])
        thumb.image(video.bigthumbhd, use_column_width=True)
        stream  = video.getbest(preftype="mp4").url
    else:
        stream = config['video_path']
        
    results, fps, n_frames, dur, time_elapse = videoprocess.process_video(stream, known_ids, progress=progress)

    st.markdown('**Result**')
    cols = st.columns(3)
    cols[0].metric('FPS', value='{} FPS'.format(fps))
    cols[1].metric('Duration', value='{}s'.format(dur))
    cols[2].metric('Total', value='{} frames'.format(n_frames))
    st.info('Time elapsed: {}'.format(datetime.timedelta(seconds=int(time_elapse))))
    
    for person in results:
        
        st.success('{} is recognized {} times'.format(person, len(results[person])))
        
        # time_range = convert_findex_to_time(results[person], fps, config['capture_every'])
        # print('Person {} appears {} times'.format(person, len(time_range)))
        
        # for start, end in time_range:
        #     if start == end:
        #         print('At {}'.format(datetime.timedelta(seconds=round(start, 2))))
        #     else:
        #         print('From {} to {}'.format(
        #             datetime.timedelta(seconds=int(start)),
        #             datetime.timedelta(seconds=int(end))
        #         ))

    print('Done!')
    
else:
    st.warning('No video link found!')

st.markdown('---')
st.header('Add identity')
name = st.text_input('Name', placeholder="Celeb's name")
images = st.file_uploader('Images', type=["png","jpg","jpeg"], accept_multiple_files=True)

if st.button('Upload indentity'):
    if name is None:
        st.error('Name not found!')
    
    if images is not None:
        upload_dir = os.path.join(config['known_image_folder'], name)
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        for image in images:
            image_name = image.name
            path = os.path.join(upload_dir, image_name)
            
            image = Image.open(image)
            image.save(path)
            
        st.success('Save all image to {}!'.format(upload_dir))
        
        # Reset ids
        exec(open(config['reset_file_path']).read())
        st.success('Reset all ids!')
        
    else:
        st.error('No images found!')