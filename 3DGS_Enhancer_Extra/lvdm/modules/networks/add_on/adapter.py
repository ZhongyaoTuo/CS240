import torch
import torch.nn as nn
from einops import rearrange
import time

# ref : https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/adapter.py#L391
class AdapterBlock(nn.Module):
    # ResNet-like block
    def __init__(self, in_channels:int, out_channels:int, num_blocks:int, down:bool=False): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.down = down

        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.blocks = nn.ModuleList([AdapterResnetBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        if self.in_conv is not None:
            x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class AdapterResnetBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x_in):
        x = self.block1(x_in)
        x = self.act(x)
        x = self.block2(x)
        return x + x_in

class Adapter(nn.Module):
    def __init__(self, in_channels:int, channels:list[int], num_blocks:int, out_channels:int, use_linear:bool=False,
                 zero_init:bool=False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.model = nn.ModuleList([AdapterBlock(channels[0], channels[0], 1, False)] + \
            [AdapterBlock(channels[i], channels[i+1], num_blocks, True) for i in range(len(channels)-1)])

        if zero_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
            print("[INFO] : adapter ; zero init")
        print("[INFO] : adapter ; self.model = ", self.model)
        time.sleep(3)
        # exit()
        
    def forward(self, x):
        # x = (B, C, T, H, W) -> reshape (B*T, C, H, W)
        # use einops
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        x = self.conv_in(x)
        feat_list = []
        # print("[DEBUG] adapter ; after conv_in : x.shape = ", x.shape) # (B*T, 40 = channels[0], H, W)
        for block in self.model:
            x = block(x)
            # print("[DEBUG] adapter ; after block x.shape = ", x.shape)
            # x_ = rearrange(x, '(b t) c h w -> b c t h w', b=b)
            # print("[DEBUG] adapter ; after rearrange x_.shape = ", x_.shape)
            feat_list.append(x)
        return feat_list[-4:]

