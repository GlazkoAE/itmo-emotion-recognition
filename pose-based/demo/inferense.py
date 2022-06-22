import numpy as np
from tensorflow.keras.models import Sequential, load_model

from detector.detector import crop_human
from extractor import Extractor


class ArousalModel:
    def __init__(self, seq_length=30, saved_model=None):
        self.seq_length = seq_length
        self.saved_model = saved_model

        # Init both models
        self.feature_extractor = Extractor()
        self.model = load_model(self.saved_model)

        # Init LSTM inputs as zeros
        self.features = np.zeros((1, seq_length, 2048))

    def predict(self, image):
        human = crop_human(image)
        feature = np.array(self.feature_extractor.extract(img=human))
        feature = np.reshape(feature, (1, 1, feature.shape[0]))
        self.features = np.concatenate((self.features[:, 1:, :], feature), axis=1)
        predict = self.model(self.features)

        return predict.numpy()[0][0]
