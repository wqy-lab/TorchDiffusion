import math

import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        halfdim = (self.dim + 1) // 2
        expo = math.log(10000) / (halfdim - 1)
        emb = torch.exp(torch.arange(0, halfdim) * -expo)
        self.register_buffer('emb', emb)

    def forward(self, t):
        t = t.float()
        li = t.unsqueeze(1) * self.emb.unsqueeze(0)
        return torch.cat([li.sin(), li.cos()], dim=1)[:, :self.dim]


class TimeResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups if in_channels % num_groups == 0 else in_channels, in_channels)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.silu2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.silu1(h)
        h = self.conv1(h)

        time_out = self.time_mlp(time_emb)
        h = h + time_out.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = self.silu2(h)
        h = self.conv2(h)

        x = h + self.residual(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, num_groups=8):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = TimeResBlock(in_channels, out_channels, time_dim, num_groups=num_groups)
    def forward(self, x, time_emb):
        return self.conv(self.maxPool(x), time_emb)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, num_groups=8):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = TimeResBlock(in_channels, out_channels, time_dim)
    def forward(self, x, skip, time_emb):
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x, time_emb)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1, base_channels = 64, channel_mults=[1, 2, 4, 8, 16], num_groups = 8, time_dim = 256):
        super().__init__()
        layer_list = []
        channels = [base_channels * i for i in channel_mults]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim), 
            nn.Linear(time_dim, time_dim), 
            nn.SiLU(), 
            nn.Linear(time_dim, time_dim)
        )
        self.intro = TimeResBlock(in_channels, channels[0], time_dim, num_groups)
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(Down(channels[i], channels[i + 1], time_dim, num_groups))
        self.ups = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.ups.append(Up(channels[i + 1], channels[i], time_dim, num_groups))
        self.outro = nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        skipList = []
        x = self.intro(x, time_emb)
        for i in range(len(self.downs)):
            skipList.append(x)
            x = self.downs[i](x, time_emb)
        for i in range(-1, -len(self.ups)-1, -1):
            x = self.ups[i](x, skipList[i], time_emb)
        x = self.outro(x)
        return x