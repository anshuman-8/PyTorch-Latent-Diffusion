import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim:int):
        super().__init__()
        # self.time_embedding = nn.Embedding(time_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim, 4 * embedding_dim)

    def forward(self, time:torch.Tensor)->torch.Tensor:

        x = self.linear1(time)
        x = F.silu(x)
        output = self.linear2(x)
        return output # (1, 1280)
    

class SwitchSequential(nn.Sequential):

    def forward(self, x:torch.Tensor, context:torch.Tensor, time:torch.Tensor)->torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        output = x
        return output


class UpSample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv =  nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        output = self.conv(x)
        return output


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, time_size:int=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, out_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(time_size, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor, time:torch.Tensor)->torch.Tensor: # x is the feature map
        residue = x

        x = self.groupnorm_feature(x)
        x = F.silu(x)

        x = self.conv_feature(x)
        x = F.silu(x)

        time = self.linear_time(time)
        # time = time.view(-1, 320, 1, 1)
        time  = time.unsqueeze(-1).unsqueeze(-1)
        x = x + time

        x = self.groupnorm_merged(x)
        x = F.silu(x)

        x = self.conv_merged(x)

        output = x + self.residual_layer(residue)
        return output
    
    
class UNetAttentionBlock(nn.Module):
    def __init__(self, n_head:int, embedding_dim:int, context_size:int=768):
        super().__init__()
        channels = embedding_dim * n_head

        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_head, channels, context_size, in_proj_bias=False)

        self.layernorm3 = nn.LayerNorm(channels)
        self.linear_geglu1 = nn.Linear(channels,4 * channels * 2)
        self.linear_geglu1 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        last_residue = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        b, c, h, w = x.shape

        x = x.view(b, c, h*w)
        x = x.transpose(-1, -2)
        
        residue = x
        # normalisation + self attention with skip connection
        x = self.layernorm1(x)
        self.attention1(x)
        x += residue

        residue = x
        # normalisation + cross attention with skip connection
        x = self.layernorm2(x)
        self.attention2(x, context)
        x += residue

        residue = x
        # normalisation + linear + GELU + linear + skip connection
        x = self.layernorm3(x)

        x, gate = self.linear_geglu1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu2(x)
        x += residue

        x = x.transpose(-1, -2)
        x = x.view((b, c, h, w))
        x += last_residue

        output = self.conv_output(x)

        return output


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Module([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)),

            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNetResidualBlock(1280, 1280)),
            SwitchSequential(UNetResidualBlock(1280, 1280))

        ])

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280),
        )

        self.decoder = nn.Module([
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280)),

            SwitchSequential(UNetResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)),

            SwitchSequential(UNetResidualBlock(1920, 1280), UNetAttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)),

            SwitchSequential(UNetResidualBlock(960, 640), UNetAttentionBlock(8, 40), UpSample(640)),

            SwitchSequential(UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)),   
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40))       
        ])

    def forward(self, latent:torch.Tensor, context:torch.Tensor, time:torch.Tensor)->torch.Tensor:
        skip_connections = []
        for layers in self.encoder:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoder:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNet_Output(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.grp_norm1 = nn.GroupNorm(32, out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.grp_norm2 = nn.GroupNorm(32, out_channels)  # try with this double too

    def forward(self, x:torch.Tensor) ->torch.Tensor:
        x = self.grp_norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # x = self.grp_norm2(x)
        # x = F.silu(x)
        # x = self.conv2(x)

        output = x
        return output

class Diffusion:
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_Output(320, 3)

    def forward(self, latent:torch.Tensor, context:torch.Tensor, time:torch.Tensor)->torch.Tensor:
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        return output
    
