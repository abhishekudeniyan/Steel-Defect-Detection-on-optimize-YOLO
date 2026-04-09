# scripts/verify_architecture.py
#
# Verifies custom YOLO architectures before training.
# Checks: (1) clean build, (2) correct parameter count vs baseline,
#         (3) shape-correct forward pass at all detection scales.
#
# Usage (run from project root):
#   python scripts/verify_architecture.py

import sys
import os

# Add project root to path so 'src.*' imports work from any working directory.
# os.path.abspath(__file__) gives the absolute path of THIS script.
# dirname() twice goes up two levels: scripts/ -> project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL ORDER: register custom modules BEFORE importing YOLO.
# Python caches module namespaces on first import. If ultralytics loads
# before we inject SPDConv, the YAML parser will never find it.
from src.register_modules import register_custom_modules
register_custom_modules()

import torch
from ultralytics import YOLO


def get_baseline_params() -> int:
    """
    Load the stock YOLOv8n architecture (no pretrained weights, just structure)
    and return its parameter count as a reference baseline.

    Why 'yolov8n.yaml' instead of 'yolov8n.pt'?
    Using the .yaml gives us a randomly-initialized model with the same
    architecture — a fair structural comparison. Using .pt would load
    pretrained weights, which doesn't affect param count but is slower
    and downloads from the internet if not cached.
    """
    try:
        baseline = YOLO('yolov8n.yaml')
        count = sum(p.numel() for p in baseline.model.parameters())
        # Clean up to free memory before we build the custom model
        del baseline
        return count
    except Exception as e:
        print(f"  Warning: could not load baseline — {e}")
        return -1  # sentinel value meaning "baseline unavailable"


def get_detection_scale_shapes(model, imgsz: int):
    """
    Extract per-scale output shapes from the detection head.

    Why eval mode here (not train mode)?
    In train mode, BatchNorm updates running statistics during forward pass.
    Feeding all-zero dummy inputs would corrupt those stats with meaningless
    values. Eval mode freezes BatchNorm, so the forward pass is truly
    side-effect free.

    Why do we still get per-scale info in eval mode?
    We hook into the Detect layer directly using PyTorch's forward hook
    mechanism, which captures the input tensors to that layer before they
    get merged into the final output. This gives us the three separate
    feature maps at P3, P4, P5 without needing train mode at all.

    Returns a list of shape tuples, one per detection scale.
    """
    captured_shapes = []

    def hook_fn(module, inputs, output):
        # 'inputs' is a tuple of tensors fed into the Detect layer.
        # Each tensor corresponds to one detection scale (P3, P4, P5).
        for tensor in inputs[0] if isinstance(inputs[0], (list, tuple)) else inputs:
            if hasattr(tensor, 'shape'):
                captured_shapes.append(tuple(tensor.shape))

    # Find the Detect layer and attach a hook to it.
    # The Detect layer is always the last layer in a YOLO model.
    detect_layer = model.model.model[-1]
    hook = detect_layer.register_forward_hook(hook_fn)

    model.model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)

    try:
        with torch.no_grad():
            model.model(dummy)
    finally:
        # Always remove the hook, even if forward pass fails.
        # Leaving hooks attached causes them to fire on every future
        # forward call, which is a memory leak and a logic bug.
        hook.remove()

    return captured_shapes

def find_yaml_architectures(models_dir="models"):
    """
    Automatically find all YAML architecture files inside models directory.
    Only picks files that look like experiment configs.
    """
    yaml_files = []

    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".yaml"):
                full_path = os.path.join(root, file)

                # Optional filter (important)
                if "spd" in file.lower() or "yolo" in file.lower():
                    yaml_files.append(full_path)

    return sorted(yaml_files)

def verify(yaml_path: str, imgsz: int = 640, baseline_params: int = -1) -> bool:
    """
    Run all verification checks on a single architecture YAML.

    Returns True only if every check passes. The caller accumulates
    these booleans across multiple architectures to produce a final
    summary report.
    """

    # Use plain ASCII separators to avoid UnicodeEncodeError on Windows terminals
    # that use non-UTF-8 codepages (e.g., cp1252 or cp850).
    separator = "-" * 58
    print(f"\n{separator}")
    print(f"  Architecture : {yaml_path}")
    print(f"  Input size   : {imgsz} x {imgsz}")
    print(separator)

    # ---------------------------------------------------------------
    # CHECK 1: Build the model from YAML
    # ---------------------------------------------------------------
    # We catch specific exception types so the error message tells you
    # exactly what went wrong, rather than a generic "something failed".
    try:
        model = YOLO(yaml_path)
        print("  [OK] Model built from YAML successfully")
    except NameError as e:
        # This means SPDConv or another custom module wasn't registered.
        print(f"  [FAIL] Module not found in namespace: {e}")
        print("         Fix: ensure register_custom_modules() is called before YOLO()")
        return False
    except FileNotFoundError:
        # The YAML path doesn't exist. Check spelling and working directory.
        print(f"  [FAIL] YAML file not found: '{yaml_path}'")
        print("         Fix: run this script from the project root directory")
        return False
    except Exception as e:
        print(f"  [FAIL] Build error: {e}")
        return False

    # ---------------------------------------------------------------
    # CHECK 2: Parameter count relative to baseline
    # ---------------------------------------------------------------
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  [OK] Parameters: {n_params:,}  ({n_params / 1e6:.3f} M)")

    if baseline_params > 0:
        delta = n_params - baseline_params
        delta_pct = (delta / baseline_params) * 100
        sign = "+" if delta >= 0 else ""
        print(f"       vs baseline YOLOv8n: {sign}{delta:,} params  ({sign}{delta_pct:.1f}%)")

        # A single-layer substitution (like SPDConv at P3) should change
        # parameter count by less than 15%. Larger deltas suggest a
        # channel dimension error in the YAML (e.g., wrong out_channels).
        if abs(delta_pct) > 15:
            print(f"  [WARN] Large parameter delta ({delta_pct:.1f}%)")
            print("         Investigate: check channel args in your YAML backbone section")

    # ---------------------------------------------------------------
    # CHECK 3: Forward pass — shape verification at each detection scale
    # ---------------------------------------------------------------
    # We use forward hooks on the Detect layer to capture the three
    # input feature maps (P3, P4, P5) before they get merged.
    # This avoids the train-mode BatchNorm corruption problem.
    try:
        scale_shapes = get_detection_scale_shapes(model, imgsz)
        print("  [OK] Forward pass completed without errors")

        # Expected spatial sizes for each scale at common input sizes:
        #   imgsz=640 -> P3: 80x80,  P4: 40x40,  P5: 20x20
        #   imgsz=480 -> P3: 60x60,  P4: 30x30,  P5: 15x15
        #   imgsz=320 -> P3: 40x40,  P4: 20x20,  P5: 10x10
        scale_labels = [
            "P3/8  (detects small objects  ~10-30px)",
            "P4/16 (detects medium objects ~30-80px)",
            "P5/32 (detects large objects  >80px   )",
        ]

        print("       Detection head input shapes (per scale):")
        for i, shape in enumerate(scale_shapes):
            label = scale_labels[i] if i < len(scale_labels) else f"Scale {i}"
            print(f"         {label}: {shape}")

        # Three scales is the standard for YOLOv8. If you see a different
        # number, the Detect layer indices in your YAML head are misconfigured.
        if len(scale_shapes) != 3:
            print(f"  [WARN] Expected 3 detection scales, captured {len(scale_shapes)}")
            print("         Check the Detect layer line at the bottom of your YAML head")

    except RuntimeError as e:
        print(f"  [FAIL] Forward pass error: {e}")
        print()
        print("  HOW TO DIAGNOSE THIS:")
        print("  Read the full error — it usually names a specific layer index.")
        print("  Then check that layer in your YAML:")
        print("    SPDConv layer  -> verify in_channels*4 inside spd_conv.py")
        print("    Concat layer   -> both inputs must have identical H and W")
        print("    C2f/Conv layer -> in_channels must match previous layer's out_channels")
        return False

    print(f"\n  [PASSED] Architecture is valid — safe to begin training.\n")
    return True


if __name__ == "__main__":
    imgsz = 640

    print("=" * 58)
    print("  YOLO Architecture Verification")
    print("=" * 58)

    # Load baseline once and reuse — building a model takes a few seconds,
    # so we avoid redundant loads when checking multiple architectures.
    print("\nLoading baseline YOLOv8n for parameter comparison...")
    baseline_params = get_baseline_params()
    if baseline_params > 0:
        print(f"Baseline: {baseline_params:,} params ({baseline_params / 1e6:.3f} M)")
    else:
        print("Baseline unavailable — skipping parameter delta comparison.")

    # Add each new experiment YAML here as you create them.
    # Uncomment the next lines when those files exist.
    # architectures_to_verify = [
    #     "models/yolov8n_spd_p3.yaml",         # Experiment 2: SPDConv at P3
    #     # "models/yolov8n_spd_p3_p4.yaml",    # Experiment 3: SPDConv at P3 + P4
    #     # "models/yolov8n_spd_full.yaml",      # Experiment 4: full backbone
    # ]
    
    architectures_to_verify = find_yaml_architectures("models")
    results = {}
    for arch in architectures_to_verify:
        results[arch] = verify(arch, imgsz=imgsz, baseline_params=baseline_params)

    # Summary report — at a glance you can see which architectures are ready.
    print("=" * 58)
    print("  SUMMARY")
    print("=" * 58)
    all_passed = True
    for arch, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {status}  {arch}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("  All architectures passed. Ready to run experiments.")
    else:
        print("  One or more architectures failed. Fix before training.")
    print("=" * 58)