import xml.etree.ElementTree as ET
from pathlib import Path

RAW_IMAGES_DIR = Path("data/raw/images")
RAW_ANN_DIR = Path("data/raw/annotations")


def verify_dataset():
    image_files = list(RAW_IMAGES_DIR.glob("*.jpg"))
    annotation_files = list(RAW_ANN_DIR.glob("*.xml"))

    print("\n===== DATASET VERIFICATION REPORT =====\n")

    missing_annotations = []
    invalid_boxes = []
    detected_classes = set()

    for img_path in image_files:
        xml_path = RAW_ANN_DIR / (img_path.stem + ".xml")

        if not xml_path.exists():
            missing_annotations.append(img_path.name)
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        for obj in root.findall("object"):
            class_name = obj.find("name").text.lower()
            detected_classes.add(class_name)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            if xmin >= xmax or ymin >= ymax:
                invalid_boxes.append(xml_path.name)

            if xmax > width or ymax > height:
                invalid_boxes.append(xml_path.name)

    print(f"Total Images: {len(image_files)}")
    print(f"Total Annotations: {len(annotation_files)}")
    print(f"Missing Annotations: {len(missing_annotations)}")
    print(f"Invalid Bounding Boxes: {len(set(invalid_boxes))}")
    print(f"Detected Classes: {sorted(detected_classes)}")

    print("\nVerification Complete.\n")


if __name__ == "__main__":
    verify_dataset()