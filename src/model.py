import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 48, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(48)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(48, 96, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(192)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(192 * 4 * 4, 100)
        self.fc1_bn = nn.BatchNorm2d(100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.elu(self.conv1_bn(self.conv1(x)))
        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = self.maxpool1(x)

        x = F.elu(self.conv3_bn(self.conv3(x)))
        x = F.elu(self.conv4_bn(self.conv4(x)))
        x = self.maxpool2(x)

        x = F.elu(self.conv5_bn(self.conv5(x)))
        x = F.elu(self.conv6_bn(self.conv6(x)))
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x


class MnistCNN(nn.Module):

    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(12)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 48, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(48)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(48 * 7 * 7, 100)
        self.fc1_bn = nn.BatchNorm2d(100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.elu(self.conv1_bn(self.conv1(x)))
        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = self.maxpool1(x)

        x = F.elu(self.conv3_bn(self.conv3(x)))
        x = F.elu(self.conv4_bn(self.conv4(x)))
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)

        return x




