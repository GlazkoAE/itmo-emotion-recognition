import data_loader as ld
from voice_extractor import extract_audio_from_video




if __name__ == "__main__":
    """
    if video:
        path_to_video='path_to_video'
        file_path=extract_audio_from_video(path_to_video)
    else:
        file_path='path_to_img'
"""
    int2emotion = {
        "01": "neutral_calm",
        "02": "neutral_calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    n_mels = 128
    file_path="Audio_Speech_Actors_01-24/Actor_*/*.wav"# path to folder with audio files
    target_path="img_data/"#path to save images

train_path,val_path=ld.load_data(file_path,target_path,n_mels, int2emotion)#returns the path to the training and validation data