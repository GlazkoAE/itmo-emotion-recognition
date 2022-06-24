import model
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import utils

if __name__ == "__main__":

    img_size = 64

    transform = utils.get_transforms(img_size)
    training_set = torchvision.datasets.ImageFolder(
        root="data_with_train_test/train/", transform=transform
    )

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=4, shuffle=True, num_workers=2
    )

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

    # net = model.Net()
    model = torchvision.models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):

        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            # print(labels," ", outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    utils.metric_for_each_class(training_loader, model, classes)
    print("Finished Training")
    PATH = "./my_resnet50_pretrained.pth"
    torch.save(model.state_dict(), PATH)
