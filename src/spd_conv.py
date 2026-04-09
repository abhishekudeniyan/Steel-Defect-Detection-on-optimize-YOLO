import torch
import torch.nn as nn


class SPD(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        s = self.stride
        B, C, H, W = x.shape

        if H % s != 0 or W % s != 0:
            x = x[..., :H - (H % s), :W - (W % s)]

        return torch.cat(
            [x[..., i::s, j::s] for i in range(s) for j in range(s)],
            dim=1
        )


class SPDConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.spd = SPD(stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

   
    
    def forward(self, x):
        # print("SPD input:", x.shape)
        x = self.spd(x)
        # print("After SPD:", x.shape)
        x = self.conv(x)
        # print("After Conv:", x.shape)
        return x

class SPDConvK3(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.spd = SPD(stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=3, stride=1 , padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

   
    
    def forward(self, x):
        print("SPD input:", x.shape)
        x = self.spd(x)
        print("After SPD:", x.shape)
        x = self.conv(x)
        print("After Conv:", x.shape)
        return x