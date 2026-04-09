import csv
from pathlib import Path


def log_metrics(experiment_name, metrics_dict):

    results_path = Path("results/metrics")
    results_path.mkdir(parents=True, exist_ok=True)

    exp_file = results_path / f"{experiment_name}.csv"
    master_file = results_path / "all_experiments.csv"

    metrics_dict["experiment"] = experiment_name

    fieldnames = list(metrics_dict.keys())

    def write_csv(csv_file):

        file_exists = csv_file.exists()

        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics_dict)

    write_csv(exp_file)
    write_csv(master_file)