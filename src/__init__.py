# src package
"""
Steel Defect Detection - Custom Modules Package

This package contains custom PyTorch modules and utilities for YOLO model extensions.
"""

# Import custom modules
from .spd_conv import SPDConv, SPD2, SPDHybrid, SPDHybrid_NO_Fuse, DKStem
from .custom_modules import M_C3k2, WeightedConcat, HybridSPDConv_3

# Import utilities
from .logger import log_metrics

# Register custom modules with Ultralytics YOLO
from .register_modules import register_custom_modules
register_custom_modules()

# Public API
__all__ = [
    # SPD modules
    "SPDConv",

    "SPD2",
    "SPDHybrid",
    "SPDHybrid_NO_Fuse",
    "DKStem",
    # Custom modules
    "M_C3k2",
    "WeightedConcat",
    "HybridSPDConv_3",
    # Utilities
    "log_metrics",
]
