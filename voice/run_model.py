import shutil

import torch.utils.data
import torchvision

from voice import utils


class ModelResnetForAudio:
    def __init__(
        self,
    ):
        pass

    def predict(self, video_name):

        # print( "RUN",video_name)
        classes = (
            "angry",
            "disgust",
            "fearful",
            "happy",
            "neutral_calm",
            "sad",
            "surprised",
        )
        num_classes = len(classes)
        path_to_weights = "./voice/my_net_only_all_dataset_resnet_50_pretrained.pth"

        img_size = 64
        transform = utils.get_transforms(img_size)
        path_to_data = utils.from_video_to_audio(video_name)

        valid_set = torchvision.datasets.ImageFolder(
            root=path_to_data, transform=transform
        )
        validation_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=4, shuffle=False, num_workers=2
        )

        # model = model.Net()
        # model.load_state_dict(torch.load(path_to_weights))
        model = torchvision.models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        model.load_state_dict(torch.load(path_to_weights))

        # utils.overall_accuracy(validation_loader, model)  # выводит общую точность
        # utils.accuracy_for_each_class(validation_loader, model, classes)  # выводит точность по классам
        predicted = utils.get_predict(
            validation_loader, model, classes
        )  # возвращает предсказанные эмоции
        shutil.rmtree(path_to_data)
        return predicted
