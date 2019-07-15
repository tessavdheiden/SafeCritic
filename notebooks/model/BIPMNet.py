import torch
import torch.nn as nn
from torch.nn import functional as F

class BIPMNet(nn.Module):
    def __init__(self):
        super(BIPMNet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.norm1 = nn.InstanceNorm2d(num_features=32)
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.norm2 = nn.InstanceNorm2d(num_features=64)
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.norm3 = nn.InstanceNorm2d(num_features=128)
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.norm4 = nn.InstanceNorm2d(num_features=256)


        #spatialtransf + resnet
        self.st_gan = STGAN()
        self.res_net = ResNet()

        self.deconv4_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1)
        self.denorm4 = nn.InstanceNorm2d(num_features=256)
        self.deconv3_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.denorm3 = nn.InstanceNorm2d(num_features=128)
        self.deconv2_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.denorm2 = nn.InstanceNorm2d(num_features=64)
        self.deconv1_1 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1)
        self.denorm1 = nn.InstanceNorm2d(num_features=3)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.leaky_relu(self.norm1(x))
        x = self.conv2_1(x)
        x = F.leaky_relu(self.norm2(x))
        x = self.conv3_1(x)
        x = F.leaky_relu(self.norm3(x))
        x = self.conv4_1(x)
        x = F.leaky_relu(self.norm4(x))

        for i in range(0, 7):

            x = self.st_gan(x)
            x = self.res_net(x)

        x = self.deconv4_1(x)
        x = F.leaky_relu(self.denorm4(x))
        x = self.deconv3_1(x)
        x = F.leaky_relu(self.denorm3(x))
        x = self.deconv2_1(x)
        x = F.leaky_relu(self.denorm2(x))
        x = self.deconv1_1(x)
        x = F.leaky_relu(self.denorm1(x))

        return x

class ResNet(nn.Module):
    def __init__(self, channels=256, kernel_size=3, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride)
        self.norm1 = nn.InstanceNorm2d(num_features=channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride)
        self.norm2 = nn.InstanceNorm2d(num_features=channels)

    def forward(self, x):
        x_init = x
        x = self.conv1(x)
        x = F.leaky_relu(self.norm1(input=x))
        x = self.conv2(x)
        x = self.norm2(x)
        return x + x_init

class STGAN(nn.Module):
    def __init__(self, channels=256, kernel_size=3, stride=1):
        super(STGAN, self).__init__()

    def forward(self, x):
        return x


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()

    def forward(self, x):
        return x

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()

    def forward(self, x):
        return x



