# src/register_modules.py

import ultralytics.nn.modules as _ultralytics_modules
import ultralytics.nn.tasks as _tasks
from src.spd_conv import SPD, SPDConv, SPDConvK3


def register_custom_modules():
    """
    Register SPD and SPDConv with Ultralytics
    WITHOUT patching parse_model
    """

    # Inject into modules namespace
    _ultralytics_modules.SPD = SPD
    _ultralytics_modules.SPDConv = SPDConv
    _ultralytics_modules.SPDConvK3 = SPDConvK3

    # Inject into tasks namespace (for eval)
    _tasks.SPD = SPD
    _tasks.SPDConv = SPDConv
    _tasks.SPDConvK3 = SPDConvK3

    print("[register_modules] SPDConv registered successfully.")


# Call the registration function when module is imported
register_custom_modules()