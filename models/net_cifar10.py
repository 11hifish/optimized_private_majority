# code from: https://github.com/jameschengpeng/PyTorch-CNN-on-CIFAR10/blob/master/ConvNetClassifier.py

from torch import nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=8 * 8 * 256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32*32*48
        x = F.relu(self.conv2(x))  # 32*32*96
        x = self.pool(x)  # 16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x))  # 16*16*192
        x = F.relu(self.conv4(x))  # 16*16*256
        x = self.pool(x)  # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8 * 8 * 256)  # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x
