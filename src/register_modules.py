# src/register_modules.py
import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import src to register custom modules BEFORE loading YOLO
import src
import pathlib 
import ultralytics.nn.modules as _ultralytics_modules
import ultralytics.nn.tasks as _tasks
from src.spd_conv import  SPDConv, SPD2,SPDHybrid,SPDHybrid_NO_Fuse,DKStem
from src.custom_modules import M_C3k2, WeightedConcat, HybridSPDConv_3


def register_custom_modules():
    """
    Register SPD and SPDConv with Ultralytics
    WITHOUT patching parse_model
    """

    # Inject into modules namespace

    # _ultralytics_modules.SPDConv = SPDConv

    _ultralytics_modules.__dict__["SPDConv"] = SPDConv

    _ultralytics_modules.__dict__["M_C3k2"] = M_C3k2
    _ultralytics_modules.__dict__["WeightedConcat"] = WeightedConcat  
    _ultralytics_modules.__dict__["SPD2"] = SPD2
    _ultralytics_modules.__dict__["SPDHybrid"] = SPDHybrid
    _ultralytics_modules.__dict__["HybridSPDConv_3"] = HybridSPDConv_3
    _ultralytics_modules.__dict__["DKStem"] = DKStem
    _ultralytics_modules.__dict__["SPDHybrid_NO_Fuse"] = SPDHybrid_NO_Fuse



    # Inject into tasks namespace (for eval)
    # _tasks.M_C3k2 = M_C3k2
    # _tasks.WeightedConcat = WeightedConcat
    # _tasks.SPDConv = SPDConv
    # _tasks.SPDConvK3 = SPDConvK3
    _tasks.__dict__["M_C3k2"] = M_C3k2
    _tasks.__dict__["WeightedConcat"] = WeightedConcat  
    _tasks.__dict__["SPDConv"] = SPDConv
    
    _tasks.__dict__["SPD2"] = SPD2
    _tasks.__dict__["SPDHybrid"] = SPDHybrid
    _tasks.__dict__["HybridSPDConv_3"] = HybridSPDConv_3
    _tasks.__dict__["DKStem"] = DKStem
    _tasks.__dict__["SPDHybrid_NO_Fuse"] = SPDHybrid_NO_Fuse



    # print("[register_modules] SPDConv registered successfully.")


# Call the registration function when module is imported
register_custom_modules()