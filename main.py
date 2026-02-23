import yaml
import argparse
from ultralytics import YOLO
from pathlib import Path
import time
import torch
from src.logger import log_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_gpu_status():
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Memory Allocated:",
              round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")
    else:
        print("Running on CPU")


def train_pipeline(config):

    start_time = time.time()

    model = YOLO(config["model"])

    results = model.train(
        data="configs/data.yaml",
        epochs=config["epochs"],
        imgsz=config["image_size"],
        batch=config["batch_size"],
        lr0=config["learning_rate"],
        optimizer=config["optimizer"],
        device=config["device"],
        patience=config["patience"],  # Early stopping
        project="experiments",
        name=config["experiment_name"],
        plots=True,
        save=True,
        workers=0,  # Avoid numpy/import errors in DataLoader worker processes (Windows + NumPy 1.x)
    )

    end_time = time.time()

    metrics = model.val()

    metrics_dict = {
        "experiment": config["experiment_name"],
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "training_time_sec": round(end_time - start_time, 2)
    }

    log_metrics(config["experiment_name"], metrics_dict)

    print("\nTraining Completed")
    print(metrics_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to training config YAML (default: configs/base.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print_gpu_status()
    train_pipeline(config)