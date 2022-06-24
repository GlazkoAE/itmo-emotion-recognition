"""
Process an image that we can pass to our networks.
"""
import cv2
import numpy as np

from pose.model.blur.anonymizer import Anonymizer

anonymizer = Anonymizer(
    method="pixel", confidence=0.5, pixel_blocks=30, gauss_kernel=3, gauss_factor=10
)


def process_image(image, target_shape, is_blur=False):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    img_arr = cv2.imread(image)
    img_arr = cv2.resize(img_arr, (w, h))

    if is_blur:
        img_arr = anonymizer.blur_face(img_arr)

    x = (img_arr / 255.0).astype(np.float32)

    return x
