import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, stride=1)
        self.norm1 = nn.InstanceNorm2d(num_features=16)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=9, stride=1)
        self.norm2 = nn.InstanceNorm2d(num_features=32)
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, stride=1)
        self.norm3 = nn.InstanceNorm2d(num_features=64)
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=9, stride=1)
        self.norm4 = nn.InstanceNorm2d(num_features=96)
        self.fc_1 = nn.Linear(in_features=3538944, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=32)
        self.fc_3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.leaky_relu(self.norm1(x))
        x = self.conv2_1(x)
        x = F.leaky_relu(self.norm2(x))
        x = self.conv3_1(x)
        x = F.leaky_relu(self.norm3(x))
        x = self.conv4_1(x)
        x = F.leaky_relu(self.norm4(x))
        x = x.view(-1, 3538944)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return torch.sigmoid(x)
