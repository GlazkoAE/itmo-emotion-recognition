from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


class FaceDetector:
    def __init__(self, w: int, h: int, configFile: str, modelFile: str):
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.w, self.h = w, h

    def __call__(self, frame: np.array, draw_bbox: bool = False) -> np.array:
        return self.find_face_box(frame, draw_bbox)

    def set_wh(self, w: int, h: int):
        self.w = w
        self.h = h

    def find_face_box(self, frame: np.array, draw_bbox: bool) -> np.array:
        self.net.setInput(
            cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
            )
        )
        bboxes = self.net.forward()
        for i in range(bboxes.shape[2]):
            confidence = bboxes[0, 0, i, 2]
            if confidence > 0.5:
                box = bboxes[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                (x, y, x1, y1) = box.astype("int")
                if (0 < y < y1 < self.h) and (0 < x < x1 < self.w):
                    face_crop = frame[y:y1, x:x1]
                    if draw_bbox:
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                else:
                    face_crop = None

                return face_crop


class ModelWrap:
    def __init__(self, jit_pth: str, purpose="face", inp_size=224):
        try:
            self.model = torch.jit.load(jit_pth)
        except Exception as e:
            print(e)
            print(
                f"Failed to load model from {jit_pth}, file not found or not cuda model."
            )
            self.model = self._through_pass

        if purpose == "face":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((inp_size, inp_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.class_names = {
                0: "Surprise",
                1: "Fear",
                2: "Disgust",
                3: "Happiness",
                4: "Sadness",
                5: "Anger",
                6: "Neutral",
            }

    def __call__(self, model_input) -> np.array:
        if model_input is None:
            return "No face detected"
        else:
            pred = self.model(
                self.transform(model_input).unsqueeze(0).to("cuda:0")
            ).argmax(axis=-1)
            return self.class_names[pred.item()]

    def _through_pass(self, out_data):
        return


class EmotionDetector:
    """Provide input image size and juse call"""

    def __init__(
        self,
        inp_img_w: int,
        inp_img_h: int,
        configFile: str = str(Path(__file__).parent / "./pretrained/deploy.prototxt"),
        modelFile: str = str(
            Path(__file__).parent
            / "./pretrained/res10_300x300_ssd_iter_140000.caffemodel"
        ),
        jit_pth: str = str(Path(__file__).parent / "./pretrained/ResnetRUL_cuda.pth"),
        inp_size: int = 224,
        draw_face_bbox=False,
    ):
        self._face_detector = FaceDetector(inp_img_w, inp_img_h, configFile, modelFile)
        self._emotion_predictor = ModelWrap(
            jit_pth=jit_pth, purpose="face", inp_size=inp_size
        )
        self._draw_face_bbox = draw_face_bbox

    def enable_face_bbox(self):
        self._draw_face_bbox = True

    def disable_face_bbox(self):
        self._draw_face_bbox = False
    
    def set_wh(self, w: int, h: int):
    	self._face_detector.set_wh(w,h)

    def __call__(self, frame: np.array) -> str:
        return self._emotion_predictor(self._face_detector(frame, self._draw_face_bbox))
