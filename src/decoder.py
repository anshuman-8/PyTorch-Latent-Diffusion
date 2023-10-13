import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.grp_norm1 = nn.GroupNorm(32,in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.grp_norm2 = nn.GroupNorm(32,out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
 
    def forward(self, x:torch.Tensor)->torch.Tensor:
        residue = x

        x=self.grp_norm1(x)
        x=F.silu(x)
        x=self.conv1(x)

        x=self.grp_norm2(x)
        x=F.silu(x)
        x=self.conv2(x)

        return x + self.residual(residue)


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.grp_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        residue = x

        n, c, h, w = x.shape

        x = x.view(n, c, h*w) # in a single layer for attention
        x = x.transpose(-1,-2) # (n, c, h*w) -> (n, h*w, c)

        x = self.attention(x)

        x = x.transpose(-1,-2)
        x = x.view(n, c, h, w)

        return x + residue

        

class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)

        )

    def forward(self, x:torch.Tensor)->torch.Tensor:

        x /= 0.18215

        for module in self:
            x = module(x)
            
        return x
