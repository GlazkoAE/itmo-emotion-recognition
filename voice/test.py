import shutil

import torch.utils.data
import torchvision
import utils

if __name__ == "__main__":
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
    # path_to_weights = './my_net_only_all_dataset_resnet_50_pretrained.pth'

    img_size = 64
    transform = utils.get_transforms(img_size)

    valid_set = torchvision.datasets.ImageFolder(
        root="data_with_train_test/val/", transform=transform
    )
    validation_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=4, shuffle=True, num_workers=2
    )

    # model = model.Net()
    # model.load_state_dict(torch.load(path_to_weights))
    model = torchvision.models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(
        torch.load("my_net_only_all_dataset_resnet_50_pretrained.pth")
    )

    # utils.overall_accuracy(validation_loader, model)  # выводит общую точность
    utils.metric_for_each_class(
        validation_loader, model, classes
    )  # выводит точность по классам
    predicted = utils.get_predict(
        validation_loader, model, classes
    )  # возвращает предсказанные эмоции
    print(predicted)
    for (img, label) in validation_loader:
        print(label)
