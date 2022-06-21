import torchvision.transforms as transforms
import torch
import torch.utils.data


def get_transforms(img_size):
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform


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


def get_predict(validation_loader,net,classes):
    full_predicted=[]
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for el in predicted:
                full_predicted.append(classes[el])
    return full_predicted


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