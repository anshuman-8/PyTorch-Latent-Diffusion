import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential): # sequence of modules to reduce the dimensionality of the input
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

            VAE_ResidualBlock(128, 128), # maybe to increase the depth of the network
            VAE_ResidualBlock(128, 128), # maybe to increase the number of parameters
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),  # height and width are halved

            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  # height and width are halved

            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  # height and width are halved

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512), # 32 is the number of groups(BatchNorm2d(512) is the same)
        )
