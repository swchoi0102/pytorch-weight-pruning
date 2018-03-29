import torch
import torchvision.transforms as transforms
import torch.optim as optim
from model import MnistCNN
import torchvision.datasets as dataset
import torch.nn as nn
from torch.autograd import Variable


def main():
    batch_size = 64

    # MNIST Dataset
    train_dataset = dataset.MNIST(root='../data/',
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)

    test_dataset = dataset.MNIST(root='../data/',
                                 train=False,
                                 transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

    net = MnistCNN()

    cuda = False
    if cuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    nepoch = 1
    display_step = 100
    for epoch in range(nepoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % display_step == display_step - 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / display_step))
                running_loss = 0.0

    print('Finished Training')

    net.eval()  # change the model to evaluation mode

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    torch.save(net.state_dict(), 'model_params.pkl')
    torch.save(optimizer.state_dict(), 'optimizer_params.pkl')
    # net.load_state_dict(torch.load('mnist_params.pkl'))


if __name__ == '__main__':
    main()
