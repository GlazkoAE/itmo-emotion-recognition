import os

import cv2
import numpy as np

import model.blur.blur as blur


class Anonymizer:
    def __init__(
        self,
        method="gauss",
        confidence=0.5,
        pixel_blocks=5,
        gauss_kernel=3,
        gauss_factor=10,
    ):
        # load serialized face detector model from disk
        print("[INFO] loading face detector model...")
        model_path = os.path.join("model", "blur", "deploy.prototxt.txt")
        weights_path = os.path.join(
            "model", "blur", "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.net = cv2.dnn.readNet(model_path, weights_path)
        self.confidence = confidence
        self.method = method
        self.pixel_blocks = pixel_blocks
        self.gauss_kernel = gauss_kernel
        self.gauss_factor = gauss_factor

    def blur_face(self, image):
        image_with_blured_face = image.copy()
        x0, y0, x1, y1 = self._get_face_box(image)
        face = image[y0:y1, x0:x1]
        blured_face = self._blur(face)
        image_with_blured_face[y0:y1, x0:x1] = blured_face
        return image_with_blured_face

    def _blur(self, image):
        if self.method == "gauss":
            return blur.anonymize_face_simple(
                image, kernel=self.gauss_kernel, factor=self.gauss_factor
            )
        elif self.method == "pixel":
            return blur.anonymize_face_pixelate(image, blocks=self.pixel_blocks)
        else:
            raise print('method must be "gauss" or "pixel"')

    def _get_face_box(self, image):
        x0, y0, x1, y1 = 0, 0, 0, 0

        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is greater
            # than the minimum confidence
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x0, y0, x1, y1) = box.astype("int")
                # extract the face ROI
                # face = image[startY:endY, startX:endX]

        return x0, y0, x1, y1
