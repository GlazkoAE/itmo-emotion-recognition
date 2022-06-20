import torch
import torchvision
import torchvision.transforms as transforms

import torch.utils.data
import model

if __name__ == '__main__':

    img_size=64
    transform = transforms.Compose(
        [transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valid_set=torchvision.datasets.ImageFolder(root='data_with_train_test/val/',transform=transform)
    validation_loader = torch.utils.data.DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=2)

    classes = ('angry','disgust','fearful','happy','neutral_calm','sad','surprised')

    PATH = './my_net.pth'


    net = model.Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    a=(False)
    a=torch.tensor(a)
    sh=a.shape

    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data

            y_true.append(labels.cpu().detach().numpy())
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.append(predicted.cpu().detach().numpy())
            c = (predicted == labels).squeeze()

            #print(c,c.shape)
            for i in range(4):
                label = labels[i]
                #print(label)
                if c.shape==sh:
                    class_correct[label] += c.item()
                else:
                    class_correct[label] += c[i].item()
                class_total[label] += 1
                if c.shape==sh:
                    break


    for i in range(7):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))