import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        # Convolution
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.norm1 = nn.InstanceNorm2d(num_features=32)
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.norm2 = nn.InstanceNorm2d(num_features=64)
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.norm3 = nn.InstanceNorm2d(num_features=128)
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.norm4 = nn.InstanceNorm2d(num_features=256)

        # Deconvolution
        self.deconv4_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1)
        self.denorm4 = nn.InstanceNorm2d(num_features=256)
        self.deconv3_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.denorm3 = nn.InstanceNorm2d(num_features=128)
        self.deconv2_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.denorm2 = nn.InstanceNorm2d(num_features=64)
        self.deconv1_1 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1)
        self.denorm1 = nn.InstanceNorm2d(num_features=3)

        # Spatial Transformer Localization-Network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3*2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=25000, out_features=32),
            nn.ReLU(True),
            nn.Linear(in_features=32, out_features=3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.res_net = ResNet()

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 25000)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(self.norm1(x))
        x = self.conv2_1(x)
        x = F.relu(self.norm2(x))
        x = self.conv3_1(x)
        x = F.relu(self.norm3(x))
        x = self.conv4_1(x)
        x = F.relu(self.norm4(x))

        for i in range(6):
            x = self.stn(x)
            x = self.res_net(x)

        x = self.deconv4_1(x)
        x = F.relu(self.denorm4(x))
        x = self.deconv3_1(x)
        x = F.relu(self.denorm3(x))
        x = self.deconv2_1(x)
        x = F.relu(self.denorm2(x))
        x = self.deconv1_1(x)
        x = F.relu(self.denorm1(x))
        return x

class ResNet(nn.Module):
    def __init__(self, channels=256, kernel_size=3, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.norm1 = nn.InstanceNorm2d(num_features=channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.norm2 = nn.InstanceNorm2d(num_features=channels)

    def forward(self, x):
        x_init = x
        x = self.conv1(x)
        x = F.relu(self.norm1(input=x))
        x = self.conv2(x)
        x = self.norm2(x)
        return x + x_init

