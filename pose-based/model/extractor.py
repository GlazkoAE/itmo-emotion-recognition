import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model


class Extractor:
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(weights="imagenet", include_top=True)

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top, so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            # self.model.layers[-1].outbound_nodes = []

        self.shape = self.model.input_shape[1:3]

    def extract_from_path(self, image_path):
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = self.extract(img)

        return features

    def extract(self, img):
        img = cv2.resize(img, self.shape)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Get the prediction.
        features = self.model.predict(img)

        return features[0]
