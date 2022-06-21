import datetime
from typing import TypedDict, NamedTuple 
import torch
import yaml
from threading import Thread
import cv2
import numpy as np
import time
from torchvision import transforms

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
        self.w = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self, (self.w, self.h)
    
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
    
    def __init__(self, w: int = 480, h: int = 360):
        self.net = cv2.dnn.readNetFromCaffe('./deploy.prototxt',
                    './res10_300x300_ssd_iter_140000.caffemodel')
        # self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.w, self.h = w, h
    
    def __call__(self, frame: np.array, draw_bbox: bool = False):
        return self.find_face_box(frame, draw_bbox)

    def set_wh(self, w: int, h: int):
        self.w = w
        self.h = h
    
    def find_face_box(self, frame: np.array, draw_bbox: bool):
        self.net.setInput(cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                                (300, 300), (104.0, 117.0, 123.0)))
        bboxes = self.net.forward()
        for i in range(bboxes.shape[2]):
            confidence = bboxes[0, 0, i, 2]
            if confidence > 0.5:
                box = bboxes[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (x, y, x1, y1) = box.astype("int") 
                if (0<y<y1<self.h) and (0<x<x1<self.w):
                    crop = frame[y:y1, x:x1]
                else:
                    break
                if draw_bbox:
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                return ((x,y,x1,y1),crop) 
        
    
class ModelPathes(NamedTuple):
    cuda_model_path: str
    cpu_model_path: str

class ModelWrap:
    
    def __init__(self, jit_pths: ModelPathes, device, purpose):
        try:
            which_model = jit_pths[0] if "cuda:0" else jit_pths[1] 
            self.model = torch.jit.load(which_model)
        except Exception as e:
            print(e)
            print(f'Failed to load model from {which_model}')
            self.model = self._through_pass
        
        if purpose == 'face':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.class_names = {0:'Surprise', 
                            1:'Fear', 
                            2:'Disgust', 
                            3:'Happiness', 
                            4:'Sadness', 
                            5:'Anger', 
                            6:'Neutral'}
        
            
    def __call__(self, model_input) -> np.array:
        pred = self.model(self.transform(model_input).unsqueeze(0).to('cuda:0')).argmax(axis=-1)
        return self.class_names[pred.item()]
    def _through_pass(self, out_data):
        return out_data

class Models(TypedDict):
    face: ModelWrap
    hand: ModelWrap
    voice: ModelWrap
