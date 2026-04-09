import sys
from pathlib import Path
# Make project root importable so `src` is found
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Register FIRST, import YOLO second ────────────────────────────────────
from src.register_modules import register_custom_modules
register_custom_modules()

import yaml
import time
import torch
import gc
from ultralytics import YOLO
from src.logger import log_metrics
import pandas as pd



def load_config():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


def run():

    cfg = load_config()

    model_name = cfg["model"]
    training = cfg["training"]
    experiment = cfg["experiment"]
    data_cfg = cfg["data"]["config"]

    seeds = experiment["seeds"]

    for seed in seeds:

        exp_name = f"{experiment['name']}_seed{seed}"

        exp_dir = Path(experiment["project_dir"]) / exp_name

        # Skip if already trained
        if exp_dir.exists():
            print(f"Skipping {exp_name} (already completed)")
            continue

        print(f"\nRunning {exp_name}\n")

        model = YOLO(model_name)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        results = model.train(
            data=data_cfg,
            imgsz=training["imgsz"],
            batch=training["batch"],
            epochs=training["epochs"],
            optimizer=training["optimizer"],
            lr0=training["lr0"],
            mosaic=training["mosaic"],
            cos_lr=training["cos_lr"],
            warmup_epochs=training["warmup_epochs"],
            patience=training["patience"],
            workers=training["workers"],
            amp=training["amp"],
            seed=seed,
            project=experiment["project_dir"],
            name=exp_name
        )

        end_time = time.time()
        training_time = end_time - start_time

        results_csv = Path(results.save_dir) / "results.csv"
        df = pd.read_csv(results_csv)

        # Column name can be either metrics/mAP50(B) (newer Ultralytics) or metrics/mAP50 (older)
        map50_col = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else "metrics/mAP50"
        best_idx = df[map50_col].idxmax()

        gpu_memory = 0

        if torch.cuda.is_available():
         gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)

        weights_path = Path(results.save_dir) / "weights/best.pt"
        model_size = weights_path.stat().st_size / (1024*1024)

        best_row = df.loc[best_idx]

        metrics = {
            
            "seed": seed,
            "mAP50": best_row.get("metrics/mAP50(B)", best_row.get("metrics/mAP50")),
            "mAP50_95": best_row.get("metrics/mAP50-95(B)", best_row.get("metrics/mAP50-95")),
            "precision": best_row.get("metrics/precision(B)", best_row.get("metrics/precision")),
            "recall": best_row.get("metrics/recall(B)", best_row.get("metrics/recall")),
            "training_time_sec": training_time,
            "gpu_memory_gb": gpu_memory,
            "model_size_mb": model_size,
            "early_stopping": best_row["epoch"] < training["epochs"],

        }

        log_metrics(experiment["name"], metrics)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    run()