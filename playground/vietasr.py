import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import make_chunks
# from tqdm import tqdm
import time
import os

# Info
filepath = r'D:\Workplaces\Python\BKC\BVideo\demo_vietasr\demo_wav\test3.wav'
chunk_length_ms = 58_000 # pydub calculates in millisec
chunk_padding = 1_000 # ms

myaudio = AudioSegment.from_file(filepath, "wav") 
chunks = make_chunks(myaudio, chunk_length_ms) # Make chunks of one sec

text_list = []
start = time.time()
for i, chunk in enumerate(chunks):
    
    print('Chunk {}'.format(i))
    
    print('Preprocessing ...')
    chunk_silent = AudioSegment.silent(duration = chunk_padding)
    audio_chunk = chunk_silent + chunk + chunk_silent
    os.makedirs('saved', exist_ok=True)
    path = os.path.join('saved', 'chunk{0}.wav'.format(i))
    audio_chunk.export(path, bitrate ='192k', format ="wav")
    
    print("Transcripting chunk {} ...".format(i))
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    try:
        rec = r.recognize_google(audio ,language="vi-VI")
        text_list.append(rec)
    except:
        text_list.append('Không xác định')
        
    print()

print('Hoàn thành trong {:.2f}s'.format(time.time() - start))
print('Kết quả: \n\n{}'.format('\n'.join(text_list)))


# import speech_recognition as sr
# import time

# r = sr.Recognizer()
# filepath = r'D:\Workplaces\Python\BKC\BVideo\demo_vietasr\demo_wav\test.wav'
# with sr.AudioFile(filepath) as source:
#     start = time.time()
#     print('Transcripting ...')
#     audio = r.record(source)
#     try:
#         text = r.recognize_google(audio,language="vi-VI")
#         print("Kết quả: {}".format(text))
#     except:
#         print("Kết quả: Không xác định")
#     print('Hoàn thành trong {.2f}s'.format(time.time() - start))