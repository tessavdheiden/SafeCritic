import torch
import torch.nn as nn

class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deconv1_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.deconv1_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.deconv1_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=3, stride=1)
        self.deconv2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel)

    def forward(self, x):
        x = torch.relu(input=self.conv1_1(x))
        x = torch.relu(input=self.conv1_2(x))
        x, indices1 = self.pool1(input=x)
        x = torch.relu(input=self.conv2_1(x))
        x = torch.relu(input=self.conv2_2(x))
        x, indices2 = self.pool2(input=x)
        x = torch.relu(input=self.conv3_1(x))
        x = torch.relu(input=self.conv3_2(x))
        x = torch.relu(input=self.conv3_3(x))
        x, indices3 = self.pool3(input=x)
        x = torch.relu(input=self.conv4_1(x))
        x = torch.relu(input=self.conv4_2(x))
        x = torch.relu(input=self.conv4_3(x))
        x, indices4 = self.pool4(input=x)

        return x