import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model import *


def main():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    net = Net()

    cuda = False
    if cuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    nepoch = 1
    display_step = 200
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

            exit()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % display_step == display_step-1:  # print every 2000 mini-batches
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

    torch.save(net.state_dict(), 'params.pkl')
    net.load_state_dict(torch.load('params.pkl'))


if __name__ == '__main__':
    main()