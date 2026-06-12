import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        layers = []
        for i in range(2):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(num_groups, out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        return self.conv(self.maxPool(x))

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1, base_channels = 64, channel_mults=[1, 2, 4, 8, 16]):
        super().__init__()
        layer_list = []
        channels = [base_channels * i for i in channel_mults]
        self.intro = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(Down(channels[i], channels[i + 1]))
        self.ups = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.ups.append(Up(channels[i + 1], channels[i]))
        self.outro = nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        skipList = []
        x = self.intro(x)
        for i in range(len(self.downs)):
            skipList.append(x)
            x = self.downs[i](x)
        for i in range(-1, -len(self.ups)-1, -1):
            x = self.ups[i](x, skipList[i])
        x = self.outro(x)
        return x