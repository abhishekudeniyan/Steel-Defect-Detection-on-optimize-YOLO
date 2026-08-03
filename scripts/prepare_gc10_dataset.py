"""
prepare_gc10_dataset.py

Convert GC10 Pascal VOC annotations to YOLO format.
- Official 10 GC10 classes
- English class names
- Fixes known annotation typos
- 80/10/10 stratified split
"""

import csv
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml


class Config:
    DATASET_ROOT = Path(r"D:\MY Projects\GC10")

    RAW_IMAGES_DIR = DATASET_ROOT / "images" / "images"
    RAW_LABELS_DIR = DATASET_ROOT / "label" / "label"

    OUTPUT_DIR = Path("data/GC10_DET_YOLO")

    TRAIN = 0.80
    VAL = 0.10
    TEST = 0.10

    RANDOM_SEED = 42
    EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


CLASS_MAPPING = {
    "1_chongkong": 0,
    "2_hanfeng": 1,
    "3_yueyawan": 2,
    "4_shuiban": 3,
    "5_youban": 4,
    "6_siban": 5,
    "7_yiwu": 6,
    "8_yahen": 7,
    "9_zhehen": 8,
    "10_yaozhe": 9,
}

CLASS_NAMES = [
    "Punching Hole",
    "Welding Line",
    "Crescent Gap",
    "Water Spot",
    "Oil Spot",
    "Silk Spot",
    "Inclusion",
    "Rolled Pit",
    "Crease",
    "Waist Folding",
]

LABEL_FIX = {
    "10_yaozhed": "10_yaozhe",
    "d": "10_yaozhe",
}

FOLDER_TO_ENGLISH = {
    "1_chongkong": "Punching Hole",
    "2_hanfeng": "Welding Line",
    "3_yueyawan": "Crescent Gap",
    "4_shuiban": "Water Spot",
    "5_youban": "Oil Spot",
    "6_siban": "Silk Spot",
    "7_yiwu": "Inclusion",
    "8_yahen": "Rolled Pit",
    "9_zhehen": "Crease",
    "10_yaozhe": "Waist Folding",
}


def log(msg):
    print(f"[INFO] {msg}")


def recreate_output(cfg):
    if cfg.OUTPUT_DIR.exists():
        shutil.rmtree(cfg.OUTPUT_DIR)
    for s in ("train","val","test"):
        (cfg.OUTPUT_DIR/"images"/s).mkdir(parents=True,exist_ok=True)
        (cfg.OUTPUT_DIR/"labels"/s).mkdir(parents=True,exist_ok=True)


def build_class_mapping():
    log("Official GC10 Classes")
    for k,v in CLASS_MAPPING.items():
        print(f"{v}: {CLASS_NAMES[v]} ({k})")
    return CLASS_MAPPING, CLASS_NAMES


def voc_to_yolo(xml_path,mapping):
    root=ET.parse(xml_path).getroot()
    w=float(root.find("size/width").text)
    h=float(root.find("size/height").text)
    out=[]

    for obj in root.findall("object"):
        cls=obj.find("name").text.strip().lower()
        cls=LABEL_FIX.get(cls,cls)

        if cls not in mapping:
            print(f"[WARNING] Unknown label '{cls}' in {xml_path.name}")
            continue

        cid=mapping[cls]
        b=obj.find("bndbox")
        xmin=max(0,float(b.find("xmin").text))
        ymin=max(0,float(b.find("ymin").text))
        xmax=min(w,float(b.find("xmax").text))
        ymax=min(h,float(b.find("ymax").text))

        xc=((xmin+xmax)/2)/w
        yc=((ymin+ymax)/2)/h
        bw=(xmax-xmin)/w
        bh=(ymax-ymin)/h

        out.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return out


def stratified_split(cfg):
    random.seed(cfg.RANDOM_SEED)
    train,val,test,stats=[],[],[],[]

    for cls_dir in sorted(cfg.RAW_IMAGES_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue

        imgs=[]
        for ext in cfg.EXTENSIONS:
            imgs.extend(cls_dir.glob(f"*{ext}"))

        random.shuffle(imgs)

        n=len(imgs)
        ntr=int(n*cfg.TRAIN)
        nva=int(n*cfg.VAL)

        tr=imgs[:ntr]
        va=imgs[ntr:ntr+nva]
        te=imgs[ntr+nva:]

        train.extend(tr)
        val.extend(va)
        test.extend(te)

        stats.append([
            FOLDER_TO_ENGLISH.get(cls_dir.name,cls_dir.name),
            n,len(tr),len(va),len(te)
        ])

    return train,val,test,stats


def process(files,split,cfg,mapping):
    log(f"Processing {split}")
    miss=0

    for img in files:
        xml=cfg.RAW_LABELS_DIR/(img.stem+".xml")

        if not xml.exists():
            miss+=1
            continue

        shutil.copy2(img,cfg.OUTPUT_DIR/"images"/split/img.name)

        labels=voc_to_yolo(xml,mapping)

        with open(cfg.OUTPUT_DIR/"labels"/split/(img.stem+".txt"),"w") as f:
            f.write("\n".join(labels))

    if miss:
        log(f"{miss} images skipped (missing XML).")


def save_statistics(cfg,stats):
    csv_file=cfg.OUTPUT_DIR/"dataset_statistics.csv"
    with open(csv_file,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["Class","Total","Train","Validation","Test"])
        w.writerows(stats)
    log(f"Statistics saved -> {csv_file}")


def generate_yaml(cfg,names):
    data={
        "path":str(cfg.OUTPUT_DIR.resolve()),
        "train":"images/train",
        "val":"images/val",
        "test":"images/test",
        "nc":10,
        "names":{i:n for i,n in enumerate(names)}
    }

    with open(cfg.OUTPUT_DIR/"data.yaml","w") as f:
        yaml.dump(data,f,sort_keys=False)

    log("data.yaml created")


def main():
    cfg=Config()

    recreate_output(cfg)
    mapping,names=build_class_mapping()

    train,val,test,stats=stratified_split(cfg)

    print("\nDataset Split")
    print("----------------------------")
    print("Train :",len(train))
    print("Val   :",len(val))
    print("Test  :",len(test))
    print("----------------------------")

    process(train,"train",cfg,mapping)
    process(val,"val",cfg,mapping)
    process(test,"test",cfg,mapping)

    save_statistics(cfg,stats)
    generate_yaml(cfg,names)

    log("Finished Successfully")


if __name__=="__main__":
    main()
