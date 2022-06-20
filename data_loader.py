import splitfolders
import shutil
import voice_extractor as extractor


def load_data(file_path="Audio",target_path='img_data',n_mels=128, int2emotion={"04": "sad", "03": "happy"}):

    extractor.convert_audio_to_img(file_path,target_path,n_mels,int2emotion)
    #print(folder_with_image)
    folder_with_image = target_path
    target_with_train_test_folder = "./data_with_train_test"
    splitfolders.ratio(folder_with_image, output=target_with_train_test_folder, seed=1337, ratio=(.9, .1))

    shutil.rmtree(target_path)


    return "data/train","data/val"

