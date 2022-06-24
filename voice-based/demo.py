from run_model import ModelResnetForAudio

if __name__ == '__main__':




    #path_to_weights = './voice-based/my_net_only_all_dataset_resnet_50_pretrained.pth'
    #path_video = './voice-based/video'
    model = ModelResnetForAudio()
    predicted = model.predict()

    print(predicted)
