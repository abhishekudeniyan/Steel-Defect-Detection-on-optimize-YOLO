
import os
import shutil
import random
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path

# ==========================================
# CONFIGURATION (Central Control)
# ==========================================

class Config:
    RAW_IMAGES_DIR = Path("data/raw/images")
    RAW_ANN_DIR = Path("data/raw/annotations")
    PROCESSED_DIR = Path("data/processed_v2")

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    RANDOM_SEED = 42
    IMAGE_EXT = "*.jpg"


# ==========================================
# LOGGER (Clean Debugging)
# ==========================================

def log(msg):
    print(f"[INFO] {msg}")


# ==========================================
# CLEAN + DIRECTORY SETUP
# ==========================================

def clean_processed_directory(cfg):
    if cfg.PROCESSED_DIR.exists():
        log("Cleaning old processed directory...")
        shutil.rmtree(cfg.PROCESSED_DIR)


def create_directory_structure(cfg):
    for split in ["train", "val", "test"]:
        (cfg.PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (cfg.PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


# ==========================================
# CLASS DETECTION
# ==========================================

def build_class_mapping(cfg):
    log("Detecting classes from XML annotations...")

    classes = set()

    for xml_file in cfg.RAW_ANN_DIR.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text.lower()
            classes.add(class_name)

    classes = sorted(list(classes))
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}

    log("Detected Classes:")
    for k, v in class_mapping.items():
        print(f"  {v}: {k}")

    return class_mapping, classes


# ==========================================
# VOC → YOLO CONVERSION
# ==========================================

def convert_voc_to_yolo(xml_path, class_mapping):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_width = float(size.find("width").text)
    img_height = float(size.find("height").text)

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower()

        if class_name not in class_mapping:
            continue

        class_id = class_mapping[class_name]

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Clamp values (important!)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)

        # Convert
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_annotations


# ==========================================
# DATA SPLIT (70/20/10)
# ==========================================

def split_dataset(image_files, cfg):
    random.seed(cfg.RANDOM_SEED)
    random.shuffle(image_files)

    total = len(image_files)

    train_end = int(total * cfg.TRAIN_RATIO)
    val_end = train_end + int(total * cfg.VAL_RATIO)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    log(f"Total Images: {total}")
    log(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    return train_files, val_files, test_files


# ==========================================
# PROCESS SPLIT
# ==========================================

def process_split(files, split_name, cfg, class_mapping):
    log(f"Processing {split_name} set...")

    for img_path in files:
        xml_path = cfg.RAW_ANN_DIR / (img_path.stem + ".xml")

        if not xml_path.exists():
            print(f"[WARNING] Missing annotation for {img_path.name}")
            continue

        # Copy image
        shutil.copy(
            img_path,
            cfg.PROCESSED_DIR / "images" / split_name / img_path.name
        )

        # Convert annotation
        yolo_labels = convert_voc_to_yolo(xml_path, class_mapping)

        label_path = (
            cfg.PROCESSED_DIR
            / "labels"
            / split_name
            / (img_path.stem + ".txt")
        )

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))


# ==========================================
# YAML GENERATOR (YOLO TRAINING READY)
# ==========================================

def generate_yaml(cfg, classes):
    yaml_data = {
        "path": str(cfg.PROCESSED_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes
    }

    yaml_path = cfg.PROCESSED_DIR / "data.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    log(f"Generated YOLO YAML at: {yaml_path}")


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    cfg = Config()

    log("===== DATASET PREPARATION STARTED =====")

    clean_processed_directory(cfg)
    create_directory_structure(cfg)

    class_mapping, classes = build_class_mapping(cfg)

    image_files = list(cfg.RAW_IMAGES_DIR.glob(cfg.IMAGE_EXT))

    if len(image_files) == 0:
        log("No images found!")
        return

    train_files, val_files, test_files = split_dataset(image_files, cfg)

    for split_name, files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        process_split(files, split_name, cfg, class_mapping)

    generate_yaml(cfg, classes)

    log("===== DATASET PREPARATION COMPLETED =====")


if __name__ == "__main__":
    main()

