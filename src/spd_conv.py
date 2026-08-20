
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


class DKStem(nn.Module):
    """
    ------------------------------------------------------------------
    Dual-Kernel Stem (Improved Version)

    Branch A:
        Conv(3x3, stride=2)
        BN
        SiLU

    Branch B:
        DWConv(5x5 , stride=2)
        BN
        SiLU

        DWConv(5x5, stride=1)
        BN
        SiLU

        PWConv(1x1)
        BN
        SiLU

    Fusion:
        Concat
        PWConv(1x1)
        BN
        SiLU
    ------------------------------------------------------------------
    """

    def __init__(self, c1, c2, debug=False):
        super().__init__()

        self.debug = debug
        c_half = c2 // 2

        # ---------------------------------------------------
        # Branch 1 (Local Features)
        # ---------------------------------------------------
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                c1,
                c_half,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(c_half),
            nn.SiLU(inplace=True)
        )

        # ---------------------------------------------------
        # Branch 2 (Large Receptive Field)
        # ---------------------------------------------------
        self.branch2 = nn.Sequential(

            # 5x5 DWConv (Downsample)
            nn.Conv2d(
                c1,
                c1,
                kernel_size=5,
                stride=2,
                padding=2,
                groups=c1,
                bias=False
            ),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),

            # 5x5 DWConv
            nn.Conv2d(
                c1,
                c1,
                kernel_size=5,
                stride=1,
                padding=2,
                groups=c1,
                bias=False
            ),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),

            # Pointwise
            nn.Conv2d(
                c1,
                c_half,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(c_half),
            nn.SiLU(inplace=True)
        )

        # ---------------------------------------------------
        # Feature Fusion
        # ---------------------------------------------------
        self.fuse = nn.Sequential(
            nn.Conv2d(
                c2,
                c2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):

        if self.debug:
            print("\n=========== DKStem ===========")
            print(f"Input        : {x.shape}")

        # Branch A
        b1 = self.branch1(x)

        if self.debug:
            print(f"Branch1 3×3  : {b1.shape}")

        # Branch B
        b2 = self.branch2(x)

        if self.debug:
            print(f"Branch2 5×5  : {b2.shape}")

        # Concatenate
        out = torch.cat((b1, b2), dim=1)

        if self.debug:
            print(f"Concat       : {out.shape}")

        # Fuse
        out = self.fuse(out)

        if self.debug:
            print(f"Output       : {out.shape}")
            print("==============================\n")

        return out

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
class SPDHybrid_old(nn.Module):
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

class SPDHybrid_NO_Fuse_old(nn.Module):
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

import torch
import torch.nn as nn

# --------------------------------------------------
# SPD Hybrid Block (with 50/50 Channel Split)
# --------------------------------------------------
class SPDHybrid(nn.Module):
    def __init__(self, c1, c2, debug=False):
        super().__init__()

        self.debug = debug
        self.spd = SPD2(stride=2, debug=debug)

        # ---- Channel Dimensions ----
        c_mid = c1 * 4        # after SPD: 4 * C_in
        c_split = c_mid // 2  # 50/50 split: 2 * C_in per branch
        c_half = c2 // 2      # C_out / 2 per branch

        # -------- Branch 1 (Pointwise 1x1) --------
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_split, c_half, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

        # -------- Branch 2 (Depthwise 3x3 + Pointwise 1x1) --------
        self.branch2 = nn.Sequential(
            nn.Conv2d(c_split, c_split, kernel_size=3, padding=1, groups=c_split, bias=False),
            nn.BatchNorm2d(c_split),
            nn.SiLU(),

            nn.Conv2d(c_split, c_half, kernel_size=1, bias=False),
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
        if self.debug:
            print("\n===== SPDHybrid Forward =====")
            print(f"[Input] {x.shape}")

        # ---- Step 1: SPD ----
        x = self.spd(x)  # [B, 4*c1, H/2, W/2]
        if self.debug:
            print(f"[SPD Output] {x.shape}")

        # ---- Step 2: 50/50 Channel Split ----
        c_split = x.shape[1] // 2
        x1, x2 = torch.split(x, c_split, dim=1)  # Two [B, 2*c1, H/2, W/2] tensors
        if self.debug:
            print(f"[Split] x1: {x1.shape}, x2: {x2.shape}")

        # ---- Step 3: Branches ----
        b1 = self.branch1(x1)  # [B, c2/2, H/2, W/2]
        b2 = self.branch2(x2)  # [B, c2/2, H/2, W/2]
        if self.debug:
            print(f"[Branch1: 1x1] {b1.shape}")
            print(f"[Branch2: DW+PW] {b2.shape}")

        # ---- Step 4: Concat & Fuse ----
        out = torch.cat([b1, b2], dim=1)  # [B, c2, H/2, W/2]
        out = self.fuse(out)              # [B, c2, H/2, W/2]

        if self.debug:
            print(f"[Fuse Output] {out.shape}")
            print("=================================\n")

        return out


# --------------------------------------------------
# SPD Hybrid Block (No Fusion Layer)
# --------------------------------------------------
class SPDHybrid_NO_Fuse(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.spd = SPD2(stride=2)

        c_mid = c1 * 4        # after SPD: 4 * C_in
        c_split = c_mid // 2  # 50/50 split: 2 * C_in per branch
        c_half = c2 // 2      # C_out / 2 per branch

        # Branch 1 (Pointwise)
        self.branch1 = nn.Sequential(
            nn.Conv2d(c_split, c_half, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

        # Branch 2 (Spatial DW 3x3 + PW 1x1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(c_split, c_split, kernel_size=3, padding=1, groups=c_split, bias=False),
            nn.BatchNorm2d(c_split),
            nn.SiLU(),

            nn.Conv2d(c_split, c_half, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.spd(x)
        c_split = x.shape[1] // 2
        x1, x2 = torch.split(x, c_split, dim=1)

        b1 = self.branch1(x1)
        b2 = self.branch2(x2)

        return torch.cat([b1, b2], dim=1)