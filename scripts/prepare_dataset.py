import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ===============================
# CONFIGURATION
# ===============================

RAW_IMAGES_DIR = Path("data/raw/images")
RAW_ANN_DIR = Path("data/raw/annotations")

PROCESSED_DIR = Path("data/processed")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ===============================
# UTILITY FUNCTIONS
# ===============================

def clean_processed_directory():
    """
    Remove existing processed directory to ensure clean reproducibility.
    """
    if PROCESSED_DIR.exists():
        print("Cleaning existing processed directory...")
        shutil.rmtree(PROCESSED_DIR)


def create_directory_structure():
    """
    Create required YOLO folder structure.
    """
    for split in ["train", "val"]:
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def build_class_mapping():
    """
    Automatically detect classes from XML annotations.
    """
    print("Detecting classes from annotations...")
    classes = set()

    for xml_file in RAW_ANN_DIR.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text.lower()
            classes.add(class_name)

    classes = sorted(list(classes))
    class_mapping = {cls_name: idx for idx, cls_name in enumerate(classes)}

    print("Detected Classes:")
    for k, v in class_mapping.items():
        print(f"{v}: {k}")

    return class_mapping


def convert_voc_to_yolo(xml_path, class_mapping):
    """
    Convert single Pascal VOC XML file to YOLO format.
    """
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

        # Safety checks
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_annotations


def split_dataset(image_files):
    """
    Split dataset into train and validation sets.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    split_index = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"\nTotal Images: {len(image_files)}")
    print(f"Train Images: {len(train_files)}")
    print(f"Validation Images: {len(val_files)}")

    return train_files, val_files


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    print("\n===== STEEL DEFECT DATASET PREPARATION =====\n")

    # Step 1: Clean previous processed directory
    clean_processed_directory()

    # Step 2: Create directory structure
    create_directory_structure()

    # Step 3: Build class mapping automatically
    class_mapping = build_class_mapping()

    # Step 4: Collect images
    image_files = list(RAW_IMAGES_DIR.glob("*.jpg"))

    if len(image_files) == 0:
        print("No images found in raw/images/")
        return

    # Step 5: Train/Val split
    train_files, val_files = split_dataset(image_files)

    # Step 6: Process each split
    for split_name, files in [("train", train_files), ("val", val_files)]:
        print(f"\nProcessing {split_name} set...")

        for img_path in files:
            xml_path = RAW_ANN_DIR / (img_path.stem + ".xml")

            if not xml_path.exists():
                print(f"Warning: Missing annotation for {img_path.name}")
                continue

            # Copy image
            shutil.copy(
                img_path,
                PROCESSED_DIR / "images" / split_name / img_path.name
            )

            # Convert annotation
            yolo_labels = convert_voc_to_yolo(xml_path, class_mapping)

            # Write label file
            label_path = (
                PROCESSED_DIR
                / "labels"
                / split_name
                / (img_path.stem + ".txt")
            )

            with open(label_path, "w") as f:
                f.write("\n".join(yolo_labels))

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()