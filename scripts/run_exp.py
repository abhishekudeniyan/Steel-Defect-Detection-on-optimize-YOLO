
import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Register custom modules FIRST
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

        if exp_dir.exists():
            print(f"Skipping {exp_name} (already completed)")
            continue

        print(f"\nRunning {exp_name}\n")

        model = YOLO(model_name)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ======================
        # TRAINING
        # ======================
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
        training_time_min = training_time / 60

        # ======================
        # LOAD TRAIN RESULTS
        # ======================
        results_csv = Path(results.save_dir) / "results.csv"
        df = pd.read_csv(results_csv)

        map50_col = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else "metrics/mAP50"
        best_idx = df[map50_col].idxmax()
        best_row = df.loc[best_idx]

        # ======================
        # GPU + MODEL SIZE
        # ======================
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)

        weights_path = Path(results.save_dir) / "weights/best.pt"
        model_size = weights_path.stat().st_size / (1024 * 1024)

        # ======================
        # VALIDATION (for speed + FPS)
        # ======================
        val_results = model.val(
            data=data_cfg,
            imgsz=training["imgsz"],
            batch=training["batch"],
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )

        speed = val_results.speed

        preprocess = speed.get("preprocess", 0)
        inference = speed.get("inference", 0)
        postprocess = speed.get("postprocess", 0)

        total_time = preprocess + inference + postprocess

        fps = 0
        if total_time > 0:
            fps = 1000 / total_time

        # ======================
        # MAIN METRICS LOG
        # ======================
        metrics = {
            "seed": seed,
            "mAP50": best_row.get("metrics/mAP50(B)", best_row.get("metrics/mAP50")),
            "mAP50_95": best_row.get("metrics/mAP50-95(B)", best_row.get("metrics/mAP50-95")),
            "precision": best_row.get("metrics/precision(B)", best_row.get("metrics/precision")),
            "recall": best_row.get("metrics/recall(B)", best_row.get("metrics/recall")),

            # existing
            "training_time_sec": training_time,

            # new
            "training_time_min": training_time_min,

            "gpu_memory_gb": gpu_memory,
            "model_size_mb": model_size,
            "early_stopping": best_row["epoch"] < training["epochs"],

            # speed
            "preprocess_ms": preprocess,
            "inference_ms": inference,
            "postprocess_ms": postprocess,
            "fps": fps,
        }

        log_metrics(experiment["name"], metrics)

        # ======================
        # TEST CSV (separate)
        # ======================
        test_metrics = {
            "seed": seed,
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "inference_ms": inference,
            "fps": fps,
        }

        test_csv_path = Path("results/metrics/test_results.csv")

        if test_csv_path.exists():
            test_df = pd.read_csv(test_csv_path)
            test_df = pd.concat([test_df, pd.DataFrame([test_metrics])], ignore_index=True)
        else:
            test_df = pd.DataFrame([test_metrics])

        test_df.to_csv(test_csv_path, index=False)

        # ======================
        # CLEANUP
        # ======================
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    run()

