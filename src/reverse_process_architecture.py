import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# turn timestep into a vector embedding that the main U-NET can use
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(), # swish activation fn
            nn.Linear(dim*4, dim)
        )

    def forward(self, t):
        # half sine and half cos waves
        half = self.dim//2

        freqs = torch.exp(-math.log(10000)* torch.arange(0, half, device=t.device) / half)

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)


        # build embeddings
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(embeddings)

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels) 

        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):

        h = self.conv1(x)
        t_out = self.time_proj(t_emb)
        h = h + t_out[:, :, None, None]
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_dim=256):
        super().__init__()

        self.time_mlp = TimeEmbedding(time_dim)

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)

        # bottleneck
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        # upsampling
        self.upsample3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        self.up3 = ResBlock(base_channels * 4 + base_channels * 4, base_channels * 2, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock(base_channels * 2 + base_channels * 2, base_channels, time_dim)

        self.upsample1 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.up1 = ResBlock(base_channels + base_channels, base_channels, time_dim)

        # output
        self.conv_out = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)  

        # down
        x0 = self.conv_in(x)           
        d1 = self.down1(x0, t_emb) 
        x_d1 = self.downsample1(d1)

        d2 = self.down2(x_d1, t_emb)
        x_d2 = self.downsample2(d2)

        d3 = self.down3(x_d2, t_emb)
        x_d3 = self.downsample3(d3)

        m = self.mid1(x_d3, t_emb)
        m = self.mid2(m, t_emb)

        # up
        u3 = self.upsample3(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3(u3, t_emb)

        u2 = self.upsample2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2, t_emb)

        u1 = self.upsample1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1(u1, t_emb)

        out = self.conv_out(u1)
        return out
