import datetime
from typing import TypedDict, NamedTuple 
import torch
import yaml
from threading import Thread
import cv2
import numpy as np
import time

def parse_cfg(cfg_path: str) -> dict:
    with open(cfg_path, 'r') as config_file:
        app_cfg = yaml.safe_load(config_file)
    return app_cfg

class FPS:
    
    def __init__(self, interval = 30):
        self._interval = interval
        self._limit = interval-1
        self._start = 0
        self._end = 0
        self._numFrames = 0
        self._fps = 0
        
    def fps(self):
        if self._numFrames%self._interval == self._limit:
            self._end = time.time()
            self._fps = 30/(self._end-self._start)
            self._start = time.time()
        self._numFrames += 1
        return self._fps
            
        
class WebcamVideoStream:
    
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True

class face_detector:
    
    def __init__(self, h=400, w=600):
        self.net = cv2.dnn.readNetFromCaffe('./res10_300x300_ssd_iter_140000.caffemodel',
                                        'deploy.prototxt.txt')
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.h, self.w = w, h
    
    def __call__(self, frame):
        return get_face_box(frame)

    def get_face_box(self, frame):
        self.net.setInput(cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                                (300, 300), (104.0, 117.0, 123.0)))
        bboxes = net.forward()
        for i in range(bboxes.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int") 
                crop = img[y:y1, x:x1]
                return ((x,y,x1,y1), crop)
            #cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        
    
class ModelPathes(NamedTuple):
    cuda_model_path: str
    cpu_model_path: str

class ModelWrap:
    
    def __init__(self, jit_pths: ModelPathes, device):
        self.model = torch.jit.load(jit_pths[0] if "cuda:0" else jit_pths[1])
    
    def __call__(self, model_input) -> np.array:
        return self.model(model_input).cpu().numpy()

class Models(TypedDict):
    face_model: ModelWrap
    hand_model: ModelWrap
    voice_model: ModelWrap
