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
            # nn.GroupNorm(32, 512), # 32 is the number of groups(BatchNorm2d(512) is the same)
            nn.BatchNorm2d(512),

            nn.SiLU(), # nn.ReLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            nn.Conv2d(8, 8, kernel_size=1, padding=0), # linear transformation to reduce the dimensionality of the input
            
        )

    def forward(self, x:torch.Tensor, noise:torch.Tensor)-> torch.Tensor:
        # x: (batch_size, 3, 512, 512)
        # noise: (batch_size, 3, 512, 512)
        for module in self:
            print(module)
            if getattr(module, 'stride', None) == (2,2): # we want to do custom asymmetrical padding
                x = F.pad(x, (0,1,0,1), value=0) # paddign only on the right side and the bottom side
            x = module(x)

        mean, log_variance = x.chunk(2, dim=1) # split the tensor into 2 tensors along the channel dimension

        # need to clamp the variance as it becomes very small
        variance = torch.exp(log_variance.clamp(min=-30, max=20)) 

        #standard deviation
        std = torch.sqrt(variance)

        """
        We have to convert a smaple from mean 0 and variance 1 to a sample from mean mu and variance sigma^2
        Z=N(0, 1) -> N(mean, variance) = mean + Z * std
        """
        x = mean + noise * std

        # scale the output
        x *= 0.18215

        return x


