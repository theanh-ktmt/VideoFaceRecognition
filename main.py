from models import VideoProcessing
from utils import convert_findex_to_time

import yaml
import os
import pickle
import pafy
import datetime
import streamlit as st

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
config['video_path'] = st.text_input('Link', value=config['video_path'])
st.markdown('---')

st.header('Conclusion')

st.markdown('**Progress**')
announce = st.progress(0)

# Get person name in video
if 'youtube' in config['video_path']:
    video = pafy.new(config['video_path'])
    best  = video.getbest(preftype="mp4").url
else:
    best = config['video_path']
    
results, fps, n_frames, dur = videoprocess.process_video(best, known_ids, progress=announce)

st.markdown('**Result**')
result_display = st.text_area(
    'Recognition result', 
    value='''
        - Video FPS:       {}
        - Video duration:  {} s
        - Video frames:    {} frame
    '''.format(
        fps, dur, n_frames
    )
)

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