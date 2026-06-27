"""
Script to verify that custom modules are properly registered with Ultralytics
"""

import ultralytics.nn.modules as ultralytics_modules
import ultralytics.nn.tasks as ultralytics_tasks


def check_registration():
    """Check if all custom modules are registered in Ultralytics"""
    
    modules_to_check = [
        "SPDConv",
        "SPDConvK3",
        "SPD2",
        "SPDHybrid",
        "SPDHybrid_NO_Fuse",
        "M_C3k2",
        "C2PSA_Lite",
        "WeightedConcat",
        "HybridSPDConv_3",
    ]
    
    print("=" * 70)
    print("ULTRALYTICS MODULE REGISTRATION CHECK")
    print("=" * 70)
    
    # Import src to trigger registration
    print("\n[1] Importing src package to trigger registration...")
    try:
        import src
        print("    ✓ src package imported successfully")
    except Exception as e:
        print(f"    ✗ Failed to import src: {e}")
        return False
    
    print("\n[2] Checking ultralytics.nn.modules namespace:")
    print("-" * 70)
    modules_ok = True
    for module_name in modules_to_check:
        if module_name in ultralytics_modules.__dict__:
            module_obj = ultralytics_modules.__dict__[module_name]
            print(f"    ✓ {module_name:<25} -> {module_obj.__module__}.{module_obj.__name__}")
        else:
            print(f"    ✗ {module_name:<25} NOT FOUND")
            modules_ok = False
    
    print("\n[3] Checking ultralytics.nn.tasks namespace:")
    print("-" * 70)
    tasks_ok = True
    for module_name in modules_to_check:
        if module_name in ultralytics_tasks.__dict__:
            module_obj = ultralytics_tasks.__dict__[module_name]
            print(f"    ✓ {module_name:<25} -> {module_obj.__module__}.{module_obj.__name__}")
        else:
            print(f"    ✗ {module_name:<25} NOT FOUND")
            tasks_ok = False
    
    print("\n" + "=" * 70)
    if modules_ok and tasks_ok:
        print("✓ ALL MODULES REGISTERED SUCCESSFULLY")
        print("=" * 70)
        return True
    else:
        print("✗ SOME MODULES NOT REGISTERED")
        print("=" * 70)
        return False


def list_all_registered_modules():
    """List all custom modules currently in Ultralytics namespaces"""
    
    print("\n" + "=" * 70)
    print("ALL REGISTERED CUSTOM MODULES IN ULTRALYTICS")
    print("=" * 70)
    
    # Import src to trigger registration
    import src
    
    # Custom module patterns to look for
    custom_patterns = ["SPD", "C2PSA", "M_C3k", "Weighted", "Hybrid"]
    
    print("\nIn ultralytics.nn.modules:")
    print("-" * 70)
    count = 0
    for name, obj in ultralytics_modules.__dict__.items():
        if any(pattern in name for pattern in custom_patterns):
            print(f"  • {name}")
            count += 1
    if count == 0:
        print("  (none found)")
    
    print("\nIn ultralytics.nn.tasks:")
    print("-" * 70)
    count = 0
    for name, obj in ultralytics_tasks.__dict__.items():
        if any(pattern in name for pattern in custom_patterns):
            print(f"  • {name}")
            count += 1
    if count == 0:
        print("  (none found)")


if __name__ == "__main__":
    # Run registration check
    success = check_registration()
    
    # List all custom modules
    list_all_registered_modules()
    
    # Test if we can import a YOLO model with custom modules
    print("\n" + "=" * 70)
    print("TESTING YOLO MODEL LOAD")
    print("=" * 70)
    try:
        from ultralytics import YOLO
        print("\n[Testing] Loading a YOLO model with custom modules...")
        model = YOLO("models/yolov8n_spd_p3.yaml")
        print("✓ Model loaded successfully!")
        print(f"  Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
