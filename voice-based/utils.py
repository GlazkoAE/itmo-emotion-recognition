import torchvision.transforms as transforms
import torch
import torch.utils.data
import glob
import librosa
import os
import skimage.io
import numpy as np
import shutil
from pydub.silence import split_on_silence
from pydub import AudioSegment
from moviepy.editor import *
import warnings



def get_transforms(img_size):
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255-img  # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


def overall_accuracy(validation_loader,net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network: %d %%' % (100 * correct / total))


def get_predict(validation_loader,model,classes):
    model.eval()
    full_predicted=[]
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            #print(images.shape,'    g',labels)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for el in predicted:
                full_predicted.append(classes[el])
    return full_predicted


def from_video_to_audio(video_path):

    os.mkdir('voice-based/audio_from_video')
    os.mkdir('voice-based/img_data')
    os.mkdir(f'voice-based/img_data/img')
    video_path = video_path+'/*.mp4'

    for file in glob.glob(video_path):
        basename = os.path.basename(file)

        target_path = "voice-based/audio_from_video/%s.wav" % basename.split('.')[0]
        video = VideoFileClip(file)
        video.audio.write_audiofile(target_path)

        newAudio = AudioSegment.from_file(target_path)
        step=3*1000
        #print(step)
        t1 = 0
        t2 = step
        temp_path=target_path.split('/')
        col=int((len(newAudio) / step))
        #print(col)

        for i,j in enumerate(range(col)):

            #print(t1,t2)
            temp_audio = newAudio[t1:t2]
            temp_audio.export(f'{temp_path[0]}/{temp_path[1]}/{temp_path[2].replace(".","")}{i}.wav', format="wav")
            t1 = t1+step
            t2 = t2+step
        newAudio.export(f'{target_path}', format="wav")
        os.remove(target_path)
        target_path=temp_path[0]+'/'+temp_path[1]+'/*.wav'


        #print(target_path,' VBVVVV')
        base_folder=from_audio_to_image(target_path)
    shutil.rmtree('voice-based/audio_from_video')
    return base_folder


def from_audio_to_image(path_audio):
    n_mels = 128
    base_folder='voice-based/img_data/'
    #print(path_audio," RRRRR")
    for files in glob.glob(path_audio):
        basename = os.path.basename(files)
        #print(basename," AAA")
        basename=basename.split('.')[0]
        #print(basename," BBB")
        basename=basename.replace('wav', '_')
        #print(basename," EEE")
        #
        #print(files," AAAAA")
        x, sr = librosa.load(files, mono=True, sr=22050)
       #
        #os.mkdir(f'img_data/img')
        #basename = basename.split('.')[0]
        out = f'voice-based/img_data/img/{basename}.png'
        #print(out)
        spectrogram_image(x, sr=sr, out=out, n_mels=n_mels)
    return base_folder




def accuracy_for_each_class(validation_loader, net, classes):
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data

            y_true.append(labels.cpu().detach().numpy())
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.append(predicted.cpu().detach().numpy())
            c = (predicted == labels).squeeze(0)

            for i in range(len(labels)):
                label = labels[i]
                if len(labels) > 1:
                    class_correct[label] += c[i].item()
                else:
                    class_correct[label] += c.item()
                    break
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))