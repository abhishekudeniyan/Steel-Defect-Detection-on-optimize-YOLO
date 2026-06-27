import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from src.spd_conv import SPD2

class M_C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(2 * c2, c2, 1)

        self.m = nn.Sequential(*[
            nn.Sequential(
                Conv(c2, c2, 2, 1),  # kernel=2 🔥
                Conv(c2, c2, 1, 1)
            ) for _ in range(n)
        ])

        self.shortcut = shortcut

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), 1))


class WeightedConcat(nn.Module):
    def __init__(self, c_out):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))
        self.conv = Conv(c_out, c_out, 1, 1)

    def forward(self, inputs):
        x1, x2 = inputs

        assert x1.shape[1] == x2.shape[1], \
            f"Channel mismatch: {x1.shape} vs {x2.shape}"

        w = torch.softmax(self.w, dim=0)
        out = w[0] * x1 + w[1] * x2

        return self.conv()

# -------------------------------
# Hybrid SPD Block
# -------------------------------
class HybridSPDConv_3(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5):
        super().__init__()

        self.spd = SPD2(stride=2)

        hidden_dim = int(in_channels * 4 * expansion)  # after SPD → 4C

        # 1x1 Conv (channel mixing)
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # Depthwise 3x3 Conv (spatial learning)
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # Final 1x1 projection
        self.pw2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Residual connection
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x

        x = self.spd(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)

        if self.use_residual:
            # Need to downsample identity to match SPD
            identity = self.spd(identity)
            x = x + identity

        return x