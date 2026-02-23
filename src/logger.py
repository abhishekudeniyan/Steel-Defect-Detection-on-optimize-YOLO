import csv
from pathlib import Path


def log_metrics(experiment_name, metrics_dict):
    results_path = Path("results/metrics")
    results_path.mkdir(parents=True, exist_ok=True)

    csv_file = results_path / "experiment_results.csv"

    file_exists = csv_file.exists()

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics_dict)
