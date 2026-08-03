import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ============================================================
# USER PATHS
# ============================================================
import os

dataset_img_path = "data\\GC10_DET_YOLO\\images"  # Update this to your actual dataset path
dataset_labels_path = "data\\GC10_DET_YOLO\\labels"  # Update this to your actual labels path
images_path = os.path.join(dataset_img_path, "train")
labels_path = os.path.join(dataset_labels_path, "train")

print("Images path:", images_path)
print("Labels path:", labels_path)
print("Images exists?", os.path.exists(images_path))
print("Labels exists?", os.path.exists(labels_path))
# ============================================================
# SETTINGS
# ============================================================
TARGET_SIZE = 640   # analyze for 640x640 input

# Head definitions for 640 input
HEADS = {
    "P1": 2,
    "P2": 4,
    "P3": 8,
    "P4": 16,
    "P5": 32
}

# Practical object-size ranges (short side in pixels at 640)
# These ranges overlap intentionally
HEAD_RANGES = {
    "P1": (4, 10),     # ultra tiny
    "P2": (8, 20),     # small
    "P3": (16, 40),    # small-medium
    "P4": (32, 80),    # medium
    "P5": (64, 9999)   # large
}

# Bins for analysis
SIZE_BINS = [
    (0, 4,   "<4 px"),
    (4, 8,   "4-8 px"),
    (8, 16,  "8-16 px"),
    (16, 32, "16-32 px"),
    (32, 64, "32-64 px"),
    (64, 128,"64-128 px"),
    (128, 9999, ">128 px")
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_all_label_files(labels_dir):
    """Find all .txt label files in YOLO format."""
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    return label_files

def parse_yolo_label_file(label_file):
    """
    Parse one YOLO label file.
    YOLO format per line: class x_center y_center width height
    Returns list of dicts.
    """
    boxes = []
    with open(label_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            w_n = float(parts[3])   # normalized width
            h_n = float(parts[4])   # normalized height
            boxes.append({
                "class_id": cls_id,
                "x_c": x_c,
                "y_c": y_c,
                "w_n": w_n,
                "h_n": h_n
            })
        except:
            continue
    return boxes

def convert_to_target_pixels(box, target_size=1024):
    """
    Since YOLO labels are normalized, for direct resize to target_size x target_size:
    width_px = w_n * target_size
    height_px = h_n * target_size
    """
    w_px = box["w_n"] * target_size
    h_px = box["h_n"] * target_size
    short_side = min(w_px, h_px)
    long_side = max(w_px, h_px)
    area = w_px * h_px
    aspect_ratio = long_side / (short_side + 1e-9)

    return {
        "class_id": box["class_id"],
        "w_px": w_px,
        "h_px": h_px,
        "short_side": short_side,
        "long_side": long_side,
        "area": area,
        "aspect_ratio": aspect_ratio
    }

def assign_size_bin(value, bins):
    """Assign value to a predefined bin."""
    for low, high, label in bins:
        if low <= value < high:
            return label
    return bins[-1][2]

def cells_covered(size_px, stride):
    """How many feature-map cells the object's short side covers."""
    return size_px / stride

def best_head_by_range(short_side):
    """
    Suggest likely best head based on practical range overlap.
    Returns list of heads whose range includes the short_side.
    """
    matched = []
    for head, (low, high) in HEAD_RANGES.items():
        if low <= short_side <= high:
            matched.append(head)
    return matched

def print_head_usability():
    print("\n" + "="*70)
    print(f"HEAD USABILITY GUIDE @ {TARGET_SIZE}x{TARGET_SIZE}")
    print("="*70)
    print("Rule of thumb based on short-side pixels:")
    print("  P1:  4–10 px")
    print("  P2:  8–20 px")
    print("  P3: 16–40 px")
    print("  P4: 32–80 px")
    print("  P5: 64+ px")
    print("\nCell coverage interpretation:")
    print("  <1 cell   : high vanish risk")
    print("  1–2 cells : weak / unstable")
    print("  2–3 cells : usable")
    print("  3–5 cells : good")
    print("  >5 cells  : strong")

# ============================================================
# MAIN ANALYSIS
# ============================================================
label_files = find_all_label_files(labels_path)

if len(label_files) == 0:
    raise FileNotFoundError(f"No label files found in: {labels_path}")

print(f"Found {len(label_files)} label files in: {labels_path}")

all_boxes = []
class_counter = Counter()
image_box_counts = []

for lf in label_files:
    boxes = parse_yolo_label_file(lf)
    image_box_counts.append(len(boxes))
    for b in boxes:
        px_box = convert_to_target_pixels(b, target_size=TARGET_SIZE)
        all_boxes.append(px_box)
        class_counter[px_box["class_id"]] += 1

if len(all_boxes) == 0:
    raise ValueError("No valid bounding boxes found in labels.")

print(f"Total bounding boxes found: {len(all_boxes)}")

# ============================================================
# EXTRACT ARRAYS
# ============================================================
short_sides = np.array([b["short_side"] for b in all_boxes], dtype=np.float32)
long_sides  = np.array([b["long_side"]  for b in all_boxes], dtype=np.float32)
widths      = np.array([b["w_px"]       for b in all_boxes], dtype=np.float32)
heights     = np.array([b["h_px"]       for b in all_boxes], dtype=np.float32)
areas       = np.array([b["area"]       for b in all_boxes], dtype=np.float32)
aspects     = np.array([b["aspect_ratio"] for b in all_boxes], dtype=np.float32)

# ============================================================
# BASIC STATS
# ============================================================
print("\n" + "="*70)
print(f"DATASET ANALYSIS @ {TARGET_SIZE}x{TARGET_SIZE}")
print("="*70)

print(f"Total images checked         : {len(label_files)}")
print(f"Total defects (all boxes)    : {len(all_boxes)}")
print(f"Avg defects / image          : {np.mean(image_box_counts):.2f}")
print(f"Max defects in one image     : {np.max(image_box_counts)}")
print(f"Min defects in one image     : {np.min(image_box_counts)}")

print("\n--- SHORT SIDE STATS (most important for small defects) ---")
print(f"Min short side (px)          : {short_sides.min():.2f}")
print(f"Max short side (px)          : {short_sides.max():.2f}")
print(f"Mean short side (px)         : {short_sides.mean():.2f}")
print(f"Median short side (px)       : {np.median(short_sides):.2f}")
print(f"P25 short side (px)          : {np.percentile(short_sides, 25):.2f}")
print(f"P75 short side (px)          : {np.percentile(short_sides, 75):.2f}")
print(f"P90 short side (px)          : {np.percentile(short_sides, 90):.2f}")

print("\n--- LONG SIDE STATS ---")
print(f"Min long side (px)           : {long_sides.min():.2f}")
print(f"Max long side (px)           : {long_sides.max():.2f}")
print(f"Mean long side (px)          : {long_sides.mean():.2f}")
print(f"Median long side (px)        : {np.median(long_sides):.2f}")

print("\n--- AREA STATS ---")
print(f"Min area (px²)               : {areas.min():.2f}")
print(f"Max area (px²)               : {areas.max():.2f}")
print(f"Mean area (px²)              : {areas.mean():.2f}")
print(f"Median area (px²)            : {np.median(areas):.2f}")

print("\n--- ASPECT RATIO STATS (long/short) ---")
print(f"Mean aspect ratio            : {aspects.mean():.2f}")
print(f"Median aspect ratio          : {np.median(aspects):.2f}")
print(f"P90 aspect ratio             : {np.percentile(aspects, 90):.2f}")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================
print("\n" + "="*70)
print("CLASS DISTRIBUTION")
print("="*70)
for cls_id, cnt in sorted(class_counter.items()):
    print(f"Class {cls_id:<3} : {cnt}")

# ============================================================
# SIZE BIN DISTRIBUTION (SHORT SIDE)
# ============================================================
size_bin_counter = Counter()
for s in short_sides:
    size_bin_counter[assign_size_bin(s, SIZE_BINS)] += 1

print("\n" + "="*70)
print(f"SHORT-SIDE SIZE DISTRIBUTION @ {TARGET_SIZE}")
print("="*70)
for _, _, label in SIZE_BINS:
    cnt = size_bin_counter[label]
    pct = (cnt / len(short_sides)) * 100
    print(f"{label:<10} : {cnt:>6} ({pct:6.2f}%)")

# ============================================================
# HEAD CELL COVERAGE ANALYSIS
# ============================================================
print("\n" + "="*70)
print("CELL COVERAGE BY HEAD (using SHORT SIDE)")
print("="*70)

head_coverage_stats = {}
for head, stride in HEADS.items():
    cov = short_sides / stride
    head_coverage_stats[head] = cov

    vanish = np.sum(cov < 1)
    weak   = np.sum((cov >= 1) & (cov < 2))
    usable = np.sum((cov >= 2) & (cov < 3))
    good   = np.sum((cov >= 3) & (cov < 5))
    strong = np.sum(cov >= 5)

    total = len(cov)
    print(f"\n{head} (stride={stride})")
    print(f"  Mean cells covered        : {cov.mean():.2f}")
    print(f"  Median cells covered      : {np.median(cov):.2f}")
    print(f"  <1 cell   (vanish risk)   : {vanish:>6} ({100*vanish/total:6.2f}%)")
    print(f"  1-2 cells (weak)          : {weak:>6} ({100*weak/total:6.2f}%)")
    print(f"  2-3 cells (usable)        : {usable:>6} ({100*usable/total:6.2f}%)")
    print(f"  3-5 cells (good)          : {good:>6} ({100*good/total:6.2f}%)")
    print(f"  >5 cells  (strong)        : {strong:>6} ({100*strong/total:6.2f}%)")

# ============================================================
# RECOMMENDED HEAD MATCHING BY PRACTICAL RANGE
# ============================================================
head_match_counter = Counter()
multi_match_examples = []

for s in short_sides:
    matched = best_head_by_range(s)
    if len(matched) == 0:
        # smaller than P1 threshold
        head_match_counter["Below P1"] += 1
    else:
        for h in matched:
            head_match_counter[h] += 1
        if len(matched) > 1:
            multi_match_examples.append((s, matched))

print("\n" + "="*70)
print("PRACTICAL HEAD-RANGE MATCH COUNTS (overlapping ranges)")
print("="*70)
for key in ["Below P1", "P1", "P2", "P3", "P4", "P5"]:
    cnt = head_match_counter.get(key, 0)
    pct = (cnt / len(short_sides)) * 100
    print(f"{key:<8} : {cnt:>6} ({pct:6.2f}%)")

# ============================================================
# AUTOMATIC RECOMMENDATION LOGIC
# ============================================================
print("\n" + "="*70)
print("AUTOMATIC RECOMMENDATION")
print("="*70)

pct_lt4   = 100 * np.sum(short_sides < 4) / len(short_sides)
pct_4_8   = 100 * np.sum((short_sides >= 4) & (short_sides < 8)) / len(short_sides)
pct_8_16  = 100 * np.sum((short_sides >= 8) & (short_sides < 16)) / len(short_sides)
pct_16_32 = 100 * np.sum((short_sides >= 16) & (short_sides < 32)) / len(short_sides)
pct_32_64 = 100 * np.sum((short_sides >= 32) & (short_sides < 64)) / len(short_sides)
pct_gt64  = 100 * np.sum(short_sides >= 64) / len(short_sides)

# Heuristic recommendation
recommendation = []
notes = []

# P1 decision
if (pct_lt4 + pct_4_8) >= 15:
    recommendation.append("P1")
    notes.append("Many ultra-small defects (<8 px) exist at 1024 → P1 should be considered.")
else:
    notes.append("Ultra-small defects are limited (<15%) → P1 may NOT be necessary initially.")

# P2 decision
if (pct_4_8 + pct_8_16 + pct_16_32) >= 10:
    recommendation.append("P2")
    notes.append("Small defects are clearly present → P2 is strongly recommended.")

# P3 decision
if (pct_8_16 + pct_16_32 + pct_32_64) >= 10:
    recommendation.append("P3")
    notes.append("Small-to-medium defects are present → P3 should be included.")

# P4 decision
if (pct_16_32 + pct_32_64 + pct_gt64) >= 10:
    recommendation.append("P4")
    notes.append("Medium-scale defects are present → P4 should be included.")

# P5 decision
if pct_gt64 >= 10:
    recommendation.append("P5")
    notes.append("A significant number of large defects (>64 px) exist → P5 is useful.")
else:
    notes.append("Large defects (>64 px) are limited → P5 may be optional.")

# Remove duplicates preserving order
recommendation = list(dict.fromkeys(recommendation))

# Fallback if empty
if len(recommendation) == 0:
    recommendation = ["P2", "P3", "P4"]
    notes.append("Fallback recommendation applied: P2-P4")

print(f"Recommended detection heads: {' + '.join(recommendation)}")
print("\nWhy:")
for n in notes:
    print(f"- {n}")

# More direct verdict
print("\nDIRECT VERDICT:")
if "P1" in recommendation:
    print("✔ P1 likely useful (many tiny defects).")
else:
    print("✔ Start WITHOUT P1. First try P2-P4.")

if "P5" in recommendation:
    print("✔ P5 useful for larger defects too.")
else:
    print("✔ P5 optional (can keep for compatibility or remove in custom design).")

if recommendation == ["P2", "P3", "P4"]:
    print("🔥 Strong likely choice for PCB @1024: P2 + P3 + P4")
elif recommendation == ["P1", "P2", "P3", "P4"]:
    print("🔥 Strong likely choice for PCB @1024: P1 + P2 + P3 + P4")
elif "P2" in recommendation and "P3" in recommendation and "P4" in recommendation:
    print("🔥 Core recommended range includes P2-P4")

# ============================================================
# PLOTS
# ============================================================
print("\n" + "="*70)
print("GENERATING PLOTS...")
print("="*70)

plt.figure(figsize=(10, 5))
plt.hist(short_sides, bins=50)
plt.title(f"Histogram of Short-Side Defect Sizes @ {TARGET_SIZE}x{TARGET_SIZE}")
plt.xlabel("Short-side size (pixels)")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(long_sides, bins=50)
plt.title(f"Histogram of Long-Side Defect Sizes @ {TARGET_SIZE}x{TARGET_SIZE}")
plt.xlabel("Long-side size (pixels)")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.show()

# Bar chart for size bins
bin_labels = [label for _, _, label in SIZE_BINS]
bin_counts = [size_bin_counter[label] for label in bin_labels]

plt.figure(figsize=(10, 5))
plt.bar(bin_labels, bin_counts)
plt.title(f"Short-Side Size Bin Distribution @ {TARGET_SIZE}x{TARGET_SIZE}")
plt.xlabel("Size bin")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.grid(True, axis='y', alpha=0.3)
plt.show()

# Cell coverage histograms for each head
for head, stride in HEADS.items():
    cov = head_coverage_stats[head]
    plt.figure(figsize=(10, 4))
    plt.hist(cov, bins=50)
    plt.title(f"{head} Cell Coverage Distribution (short-side / stride={stride})")
    plt.xlabel("Cells covered")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================
# FINAL SUMMARY BLOCK
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY (THESIS-FRIENDLY)")
print("="*70)

summary_text = f"""
For a training resolution of {TARGET_SIZE}×{TARGET_SIZE}, the dataset was analyzed using the
short-side dimension of each ground-truth bounding box, which is the most relevant measure
for tiny and elongated PCB defects. The short-side distribution indicates whether objects
remain sufficiently represented after resizing and downsampling.

Objects were mapped to detection heads using effective pixel ranges:
P1: 4–10 px, P2: 8–20 px, P3: 16–40 px, P4: 32–80 px, P5: 64+ px.

Automatic recommendation based on the observed size distribution:
{' + '.join(recommendation)}

This recommendation is derived from the proportion of defects falling into ultra-small,
small, medium, and large size intervals after resizing to {TARGET_SIZE}×{TARGET_SIZE}.
"""

print(summary_text.strip())

print_head_usability()