import cv2
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Project root: same folder that contains 'data' and 'scripts'
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
RAW_ANN_DIR = PROJECT_ROOT / "data" / "raw" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_random_samples(num_samples=10, show=True, save=True):
    image_files = list(RAW_IMAGES_DIR.glob("*.jpg"))
    if not image_files:
        print("No images found in data/raw/images/")
        return

    n = min(num_samples, len(image_files))
    selected_images = random.sample(image_files, n)
    saved = 0

    for img_path in selected_images:
        xml_path = RAW_ANN_DIR / (img_path.stem + ".xml")
        if not xml_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                image,
                class_name,
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if save:
            output_path = OUTPUT_DIR / f"vis_{img_path.name}"
            cv2.imwrite(str(output_path), image)
            saved += 1

        if show:
            cv2.imshow(img_path.name, image)
            print("Press any key for next image, or 'q' to quit early.")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    if save:
        print(f"Saved {saved} visualization(s) to results/plots/")
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_random_samples(num_samples=10, show=True, save=True)