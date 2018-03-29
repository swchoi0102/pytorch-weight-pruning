from model import MnistCNN
from pytorchpruner.modules import MaskedModule
from pytorchpruner.pruners import BasePruner
from torch.autograd import Variable
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


def main():

    net = MnistCNN()

    # load pretrained parameters
    net.load_state_dict(torch.load('model_params.pkl'))

    net = MaskedModule(net)
    pruner = BasePruner(net)

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

    cuda = False
    if cuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4)

    nepoch = {
        0.2: 1,
        0.4: 1,
        0.6: 1,
        0.8: 10,
        0.9: 10,
        0.95: 10,
    }

    for prate in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
        pruner.prune(prate)  # ex) 0.8 -> 80% are pruned
        net.train()

        display_step = 100
        for epoch in range(nepoch[prate]):
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

                net.apply_mask_on_gradients()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % display_step == display_step - 1:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / display_step))
                    running_loss = 0.0

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

        test_acc = 100 * correct / total
        print('Pruned Rate: {}%, Test acc: {}'.format(prate * 100, test_acc))
        torch.save(net.state_dict(), 'model_{}_params.pkl'.format(prate))
        # print(net)


if __name__ == '__main__':
    main()