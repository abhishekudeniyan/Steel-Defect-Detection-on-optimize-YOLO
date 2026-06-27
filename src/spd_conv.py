
import torch
import torch.nn as nn

try:
    # Older Ultralytics releases expose SPD here.
    from ultralytics.nn.modules.conv import SPD
except ImportError:
    class SPD(nn.Module):
        """Compatibility fallback when Ultralytics does not expose SPD."""

        def __init__(self, stride=2):
            super().__init__()
            self.stride = stride

        def forward(self, x):
            b, c, h, w = x.shape
            s = self.stride
            if h % s != 0 or w % s != 0:
                x = x[..., :h - (h % s), :w - (w % s)]
            return torch.cat([x[..., i::s, j::s] for i in range(s) for j in range(s)], dim=1)


class SPDConv(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        self.spd = SPD(stride=2)
        # Ultralytics may pass either (c1, c2) or only (c2) for custom layers.
        if c2 is None:
            self.c2 = c1
            self._expected_c1 = None
            self.conv = None
        else:
            self.c2 = c2
            self._expected_c1 = c1
            self.conv = self._build_conv(c1, c2, kernel_size=1)

    @staticmethod
    def _build_conv(c1, c2, kernel_size):
        pad = 0 if kernel_size == 1 else kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        if self.conv is None:
            self.conv = self._build_conv(x.shape[1], self.c2, kernel_size=1).to(x.device)
        x = self.spd(x)
        return self.conv(x)




# ----------------------------------
# SPD Layer
# ----------------------------------
class SPD2(nn.Module):
    def __init__(self, stride=2, debug=False):
        super().__init__()
        self.stride = stride
        self.debug = debug

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.stride

        if self.debug:
            print(f"[SPD] Input: {x.shape}")

        # Ensure divisible
        if H % s != 0 or W % s != 0:
            x = x[..., :H - (H % s), :W - (W % s)]

        out = torch.cat(
            [x[..., i::s, j::s] for i in range(s) for j in range(s)],
            dim=1
        )

        if self.debug:
            print(f"[SPD] Output: {out.shape}  (C→{C*4}, H→{H//2}, W→{W//2})")

        return out


# ----------------------------------
# SPD Hybrid Block
# ----------------------------------
class SPDHybrid(nn.Module):
    def __init__(self, c1, c2, debug=False):
        super().__init__()

        self.debug = debug
        self.spd = SPD2(stride=2, debug=debug)

        c_mid = c1 * 4   # after SPD
        c_half = c2 // 2

        # -------- Branch 1 (1x1) --------
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_mid, c_half, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

        # -------- Branch 2 (Depthwise 3x3 + Pointwise) --------
        self.branch2 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),

            nn.Conv2d(c_mid, c_half, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

        # -------- Fusion --------
        self.fuse = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        if self.debug:
            print("\n===== SPDHybrid Forward =====")
            print(f"[Input] {x.shape}")

        # ---- SPD ----
        x = self.spd(x)
        B, C_spd, H_spd, W_spd = x.shape

        # ---- Branch 1 ----
        b1 = self.branch1(x)
        if self.debug:
            print(f"[Branch1: 1x1] {b1.shape}  (channel mixing only)")

        # ---- Branch 2 ----
        b2 = self.branch2(x)
        if self.debug:
            print(f"[Branch2: DW+PW] {b2.shape}  (spatial + channel)")

        # ---- Concat ----
        out = torch.cat([b1, b2], dim=1)
        if self.debug:
            print(f"[Concat] {out.shape}  (c2 channels)")

        # ---- Fuse ----
        out = self.fuse(out)
        if self.debug:
            print(f"[Fuse Output] {out.shape}")
            print("=================================\n")

        return out

class SPDHybrid_NO_Fuse(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.spd = SPD(stride=2)

        c_mid = c1 * 4  # after SPD

        # split output exactly into c2
        c_half = c2 // 2

        # Branch 1 (fast)
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_mid, c_half, 1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

        # Branch 2 (spatial)
        self.branch2 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, padding=1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),

            nn.Conv2d(c_mid, c_half, 1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.spd(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)

        return torch.cat([b1, b2], dim=1)