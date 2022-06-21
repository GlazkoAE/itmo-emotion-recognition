import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import model

if __name__ == '__main__':

    img_size = 64
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_set = torchvision.datasets.ImageFolder(root='data_with_train_test/train/', transform=transform)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)

    classes = ('angry', 'disgust', 'fearful', 'happy', 'neutral_calm', 'sad', 'surprised')

    net = model.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(20):

        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            # print(labels," ", outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    PATH = './my_net.pth'
    torch.save(net.state_dict(), PATH)

    net.eval()
    net.to("cpu")
    dummy_input = torch.randn(1, 3, 64, 64)
    input_names = ["actual_input"]
    output_names = ["output"]
    torch.onnx.export(net,
                      dummy_input,
                      "SER.onnx",
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=11
                      )

