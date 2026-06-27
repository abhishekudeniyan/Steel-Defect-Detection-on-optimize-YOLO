# from ultralytics import YOLO
# import torch

# def model_summary(model):
#     total_params = 0
#     trainable_params = 0

#     print("\n🔍 Model Layer-wise Summary:\n")

#     for name, param in model.named_parameters():
#         total_params += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()

#         print(f"{name:50} | {param.shape} | {param.numel()}")

#     print("\n📊 Summary:")
#     print(f"Total Parameters: {total_params:,}")
#     print(f"Trainable Parameters: {trainable_params:,}")
#     print(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}")

# def main():
#     model = YOLO("models\yolov8n_spd_p3.yaml")  # Update with your actual model path

#     # Build model
#     model = model.model

#     print("\n✅ Model Loaded Successfully\n")

#     # Print architecture
#     print(model)

#     # Count parameters
#     model_summary(model)

# if __name__ == "__main__":
#     main()

import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import src to register custom modules BEFORE loading YOLO
import src

import torch
from ultralytics import YOLO


def analyze_model(model_path):
    model = YOLO(model_path).model
    x = torch.randn(1, 3, 640, 640)

    print("\n===== MODEL ANALYSIS =====\n")

    total_params = 0

    for i, layer in enumerate(model.model):
        params = sum(p.numel() for p in layer.parameters())
        total_params += params

        try:
            x = layer(x)
            shape = list(x.shape)
        except:
            shape = "skip (multi-input)"

        print(f"Layer {i:2d} | {layer.__class__.__name__:20s} | Params: {params:8d} | Output: {shape}")

    print("\n==========================")
    print(f"Total Parameters: {total_params:,}")
analyze_model("models/Baselineyolov8nano.yaml")