from moviepy.editor import *
import numpy as np
import skimage.io
import librosa
import glob
import librosa.display
import os
import pathlib


def extract_audio_from_video(file_path: str) -> str:

    target_path = "audio_from_video/%s.wav" % file_path.split('.')[0]
    video = VideoFileClip(file_path)
    video.audio.write_audiofile(target_path)
    return target_path


def convert_audio_to_img(file_path: str,target_path: str,n_mels: int, int2emotion: dict) :

    #print(file_path,target_path,int2emotion)

    for e in int2emotion.values():
        pathlib.Path(f'{target_path}/{e}').mkdir(parents=True, exist_ok=True)

    for file in glob.glob(file_path):

        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        x, sr = librosa.load(file, mono=True, sr=22050)
        out = f'{target_path}/{emotion}/{basename[:-3].replace(".", "")}.png'
        # print(out)
        spectrogram_image(x, sr=sr, out=out, n_mels=n_mels)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, n_mels):
    # use log-melspectrogram

    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)
