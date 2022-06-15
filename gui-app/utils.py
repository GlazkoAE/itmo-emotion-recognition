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
