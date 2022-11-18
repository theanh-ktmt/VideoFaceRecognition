# VideoFaceRecognition
Count number of celebrity's face appear on a Youtube video

### 1. Pretrained model
Model: [ONNX file](https://drive.google.com/drive/folders/1bOpQP5s2Hmc3hNA7tAzg7ENmvKS3d1wl?usp=share_link)

### 2. Installation
- Clone git repo [InsightFace](https://github.com/deepinsight/insightface)
```
git clone https://github.com/deepinsight/insightface
```
- Install requirements
```
pip install -r requirements.txt
```
- Comment line 53, 54 of backend_youtube_dl.py in pafy library in your environment
```
self._viewcount = self._ydl_info['view_count']
# self._likes = self._ydl_info['like_count']
# self._dislikes = self._ydl_info['dislike_count']
self._username = self._ydl_info['uploader_id']
```
- Download above-mentioned pretrained model and save to a folder name **saved**

### 3. Usage
- Add new folder with person name to directory dataset/images
- Run tools/create_known_ids.py to recalculate the person mean ids
- Run program with command 
```
streamlit run main.py
```
- Open browser at [http://localhost:8501/](http://localhost:8501/)
- Fill Youtube video link or video link on your computer and press Enter to run the program
- It'll take some time to process the video, please wait. When it's done, the result will be shown below


