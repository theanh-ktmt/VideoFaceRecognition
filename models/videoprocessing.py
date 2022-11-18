from .frameprocessing import FrameProcessing
import cv2
import time
import datetime

class VideoProcessing:
    def __init__(self, config):
        self.config = config
        self.frame_processing = FrameProcessing(config)
        
    def process_video(self, video_path, known_ids, progress=None):
        
        print('Processing video at {} ...'.format(video_path))
        start = time.time()
        
        vidcap = cv2.VideoCapture(video_path)
        vidfps = int(vidcap.get(cv2.CAP_PROP_FPS))
        vidframe = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        viddur =  vidframe / vidfps
        
        print('Video FPS: {}'.format(vidfps))
        print('Video duration {:.2f}s'.format(viddur))
        
        # Return result key: person name, value: list of time that person appear
        results = {}
        
        # Params
        frame_count = 0
        success = True
        
        while success:
            success, frame = vidcap.read()
            
            if frame_count % (vidfps * self.config['capture_every']) == 0:
                
                print('- Processing frame at second {} / {}'.format(int(frame_count / vidfps), int(viddur)))
                if progress is not None:
                    progress.progress(int((frame_count + 1) / vidframe * 100))
                    
                name_list = self.frame_processing.process_frame(frame, known_ids)
                for name in name_list:
                    if name not in results:
                        results[name] = [frame_count]
                    else:
                        if results[name][-1] != frame_count:
                            results[name].append(frame_count)
            
            frame_count += 1
        
        print('Done after {}!'.format(
            datetime.timedelta(seconds=int(time.time() - start))
        ))
        
        return results, vidfps, vidframe, viddur