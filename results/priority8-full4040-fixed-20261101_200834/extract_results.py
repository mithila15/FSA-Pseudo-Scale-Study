import json
import csv
import glob
import os

ROOT = "runs/priority8"
OUT = "extracted_reports"
os.makedirs(OUT, exist_ok=True)

metric_files = sorted(
    glob.glob(f"{ROOT}/*/test-four-benchmarks/metrics.json")
)

if not metric_files:
    raise FileNotFoundError("No metrics.json files found!")

all_rows = []
weighted_rows = []

for path in metric_files:
    data = json.load(open(path))
    config = os.path.basename(
        os.path.dirname(os.path.dirname(path))
    )

    datasets = data["datasets"]

    for dataset, metrics in datasets.items():
        row = {
            "Configuration": config,
            "Dataset": dataset.upper(),
            **metrics
        }
        all_rows.append(row)

    total_count = sum(m["count"] for m in datasets.values())

    aggregate = {
        "Configuration": config,
        "Total_images": total_count,
    }

    metric_names = [
        "S", "wF", "MAE",
        "E_adaptive", "E_mean", "E_max",
        "Fbeta_adaptive", "Fbeta_mean", "Fbeta_max",
        "IoU_0_5", "Dice_0_5",
        "Precision_0_5", "Recall_0_5",
        "Specificity_0_5", "BER_0_5",
        "FPR_0_5", "FNR_0_5",
        "Accuracy_0_5", "BalancedAccuracy_0_5",
        "BoundaryIoU_0_5"
    ]

    for metric in metric_names:
        aggregate[metric] = sum(
            ds["count"] * ds[metric]
            for ds in datasets.values()
        ) / total_count

    weighted_rows.append(aggregate)

def save_csv(filename, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))
    with open(os.path.join(OUT, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

save_csv("all_dataset_metrics.csv", all_rows)
save_csv("weighted_comparison.csv", weighted_rows)

# Rank configurations: higher is better except MAE
ranking_metrics = {
    "S": True,
    "wF": True,
    "MAE": False,
    "Fbeta_max": True,
    "BoundaryIoU_0_5": True,
}

rank_rows = []

for metric, higher_is_better in ranking_metrics.items():
    ordered = sorted(
        weighted_rows,
        key=lambda x: x[metric],
        reverse=higher_is_better
    )

    for rank, row in enumerate(ordered, 1):
        rank_rows.append({
            "Metric": metric,
            "Rank": rank,
            "Configuration": row["Configuration"],
            "Score": row[metric]
        })

save_csv("performance_ranking.csv", rank_rows)

print("\nExtraction completed!")
print("Configurations:", len(metric_files))
print("Dataset metric rows:", len(all_rows))
print("Total image evaluations:", sum(
    row["count"] for row in all_rows
))
print("\nReports saved in:", OUT)
for name in os.listdir(OUT):
    print(" -", name)
