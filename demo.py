import torchvision
import torch.utils.data
import model
import utils

if __name__ == '__main__':
    classes = ('angry', 'disgust', 'fearful', 'happy', 'neutral_calm', 'sad', 'surprised')
    path_to_weights = './my_net.pth'
    path_to_data = 'data_with_train_test/val/'
    img_size = 64
    transform = utils.get_transforms(img_size)

    valid_set = torchvision.datasets.ImageFolder(root=path_to_data, transform=transform)
    validation_loader = torch.utils.data.DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=2)

    net = model.Net()
    net.load_state_dict(torch.load(path_to_weights))

    utils.overall_accuracy(validation_loader, net)  # выводит общую точность
    utils.accuracy_for_each_class(validation_loader, net, classes)  # выводит точность по классам
    predicted = utils.get_predict(validation_loader, net, classes)  # возвращает предсказанные эмоции
    print(predicted)
