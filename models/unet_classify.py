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
    def __init__(self, in_channels, out_channels, time_dim, emb_size, num_groups=8, num_heads=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups if in_channels % num_groups == 0 else in_channels, in_channels)
        self.silu1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.crossAttention = CrossAttention(out_channels, emb_size, num_heads)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.silu2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, time_emb, labels=None):
        h = self.norm1(x)
        h = self.silu1(h)
        h = self.conv1(h)

        time_out = self.time_mlp(time_emb)
        h = h + time_out.unsqueeze(-1).unsqueeze(-1)

        attn = self.crossAttention(h, labels)
        h = h + attn

        h = self.norm2(h)
        h = self.silu2(h)
        h = self.conv2(h)

        x = h + self.residual(x)
        return x

class LabelsEmbedding(nn.Module):
    def __init__(self, class_labels, emb_size):
        super().__init__()
        self.null_emb = nn.Parameter(torch.zeros(1, emb_size))
        self.embedding = nn.Embedding(class_labels, emb_size)

    def forward(self, labels=None, dropout=0.0, batch_size=None):
        if labels is not None:
            if not self.training or dropout == 0.0:
                return self.embedding(labels)
            B = labels.shape[0]
            device = labels.device
            r = torch.rand(B, device=device) < dropout
            emb = self.embedding(labels)
            null_emb = self.null_emb.expand(B, -1)
            return torch.where(r.unsqueeze(1), null_emb, emb)
        if batch_size is None:
            raise ValueError('batch_size is required when labels is None')
        return self.null_emb.expand(batch_size, -1)
    

class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_size, num_heads=4):
        super().__init__()

        self.num_heads = num_heads
        while in_channels % self.num_heads != 0:
            self.num_heads -= 1
        
        self.head_dims = in_channels // self.num_heads

        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(emb_size, in_channels)
        self.to_v = nn.Linear(emb_size, in_channels)
        self.to_out = nn.Linear(in_channels, in_channels)

        self.scale = self.head_dims ** 0.5

    def forward(self, x, emb=None):
        B, C, H, W = x.shape

        x = x.view(B, C, H*W).permute(0, 2, 1) # (B, H*W, C)

        Q = self.to_q(x) # (B, H*W, C)
        K = self.to_k(emb) # (B, 1, C)
        V = self.to_v(emb) # (B, 1, C)

        Q = Q.view(B, -1, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        # (B, N, H*W, D)
        K = K.view(B, -1, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        V = V.view(B, -1, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        # (B, N, t, D)

        attn = (Q @ K.transpose(-2, -1)) * self.scale # (B, N, H*W, t)
        attn = attn.softmax(dim=-1)

        out = attn @ V # (B, N, H*W, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H*W, C)
        out = self.to_out(out)

        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return out

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, emb_size, num_groups=8, num_heads=8):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = TimeResBlock(in_channels, out_channels, time_dim, emb_size, num_groups, num_heads)
    def forward(self, x, time_emb, labels=None):
        return self.conv(self.maxPool(x), time_emb, labels)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, emb_size, num_groups=8, num_heads=8):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = TimeResBlock(in_channels, out_channels, time_dim, emb_size, num_groups, num_heads)
    def forward(self, x, skip, time_emb, labels=None):
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x, time_emb, labels)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, channel_mults, time_dim, emb_size, num_classes, num_groups=8, num_heads=8, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        channels = [base_channels * i for i in channel_mults]

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.labels_embedding = LabelsEmbedding(num_classes, emb_size)
        self.intro = TimeResBlock(in_channels, channels[0], time_dim, emb_size, num_groups, num_heads)
        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(Down(channels[i], channels[i + 1], time_dim, emb_size, num_groups, num_heads))
        self.ups = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.ups.append(Up(channels[i + 1], channels[i], time_dim, emb_size, num_groups, num_heads))
        self.outro = nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t, labels=None):
        time_emb = self.time_mlp(t)
        labels_emb = self.labels_embedding(
            labels,
            dropout=self.dropout,
            batch_size=x.size(0),
        )
        skipList = []
        x = self.intro(x, time_emb, labels_emb)
        for i in range(len(self.downs)):
            skipList.append(x)
            x = self.downs[i](x, time_emb, labels_emb)
        for i in range(-1, -len(self.ups) - 1, -1):
            x = self.ups[i](x, skipList[i], time_emb, labels_emb)
        x = self.outro(x)
        return x
