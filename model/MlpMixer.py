import sys
sys.path.append('..')

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange

from mlpmixer_cfg import *

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim):
        super().__init__()
    
        self.norm1 = nn.LayerNorm(dim)
        self.rearrange1 = Rearrange('b n d -> b d n')
        self.dense_token1 = nn.Linear(num_patches, token_dim)
        self.gelu1 = nn.GELU()
        self.dense_token2 = nn.Linear(token_dim, num_patches)
        self.rearrange2 = Rearrange('b d n -> b n d')
        self.norm2 = nn.LayerNorm(dim)
        self.dense_channel1 = nn.Linear(dim, channel_dim)
        self.gelu2 = nn.GELU()
        self.dense_channel2 = nn.Linear(channel_dim, dim)

    def forward(self, img):
        x = self.norm1(img)
        x = self.rearrange1(x)
        x = self.dense_token1(x)
        x = self.gelu1(x)
        x = self.dense_token2(x)  
        x = self.rearrange2(x)
        img = img + x
        x = self.norm2(img)
        x = self.dense_channel1(x)
        x = self.gelu2(x)
        x = self.dense_channel2(x)
        return img + x

class MLPMixer(nn.Module):
    def __init__(self, arch='B_16', image_size=224, channels=3, patch_size=16, dim=768, depth=12, num_classes=1000, token_dim=384, channel_dim=3072, training=True):
        super().__init__()
        if arch:
            if arch == 'B_16':
                configMLP = get_b16_config()
            elif arch == 'L_16':
                configMLP = get_l16_config()
            elif arch == 'B_16_imagenet1k':
                configMLP = get_b16_imagenet1k_config()
            elif arch == 'L_16_imagenet1k':
                print('yes')
                configMLP = get_l16_imagenet1k_config()

            self.image_size = configMLP['image_size']
            self.image_h, self.image_w = pair(self.image_size)
            assert (self.image_h % patch_size) == 0 and (self.image_w % patch_size) == 0, 'image must be divisible by patch size'
            self.num_patches = (self.image_h // configMLP['patch_size']) * (self.image_w // configMLP['patch_size'])
            self.dim = configMLP['dim']
            self.depth = configMLP['depth']
            self.token_dim = configMLP['token_dim']
            self.channel_dim = configMLP['channel_dim']
            self.num_classes =  configMLP['num_classes'] if training else num_classes
        else:
            self.image_size = image_size
            self.image_h, self.image_w = pair(self.image_size)
            assert (self.image_h % patch_size) == 0 and (self.image_w % patch_size) == 0, 'image must be divisible by patch size'
            self.num_patches = (self.image_h // patch_size) * (self.image_w // patch_size)
            self.dim = dim
            self.depth = depth
            self.token_dim = token_dim
            self.channel_dim = channel_dim
            self.num_classes = num_classes

        self.conv = nn.Conv2d(in_channels=channels, out_channels=self.dim, kernel_size=patch_size, stride=patch_size)
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        self.mixerblocks = nn.ModuleList([MixerBlock(self.dim, self.num_patches, self.token_dim, self.channel_dim) for _ in range(self.depth)])
        self.prenorm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.num_classes)
    def forward(self, img):
        x = self.conv(img)
        x = self.rearrange(x)
        for block in self.mixerblocks:
            x = block(x)
        x = self.prenorm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x