"""Download a small COCO val2017 sample: images + filtered GT annotations.

Run on the Pi (or anywhere). Downloads the official COCO val2017 annotation
file once (cached in data/), filters it to N images that have several
annotations across at least 2 categories, downloads those images, downscales
them to max 640px, and writes:
  - examples/gt_sample.json   (committed — the demo ground truth)
  - examples/images/*.jpg     (committed — the demo images)

The full annotation file stays in data/ (gitignored) so the filter can be
re-run with a different seed/count without re-downloading.

Usage:
  python data/download_sample.py [--count 12] [--seed 42] [--max-size 640]
"""

import argparse
import io
import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXAMPLES_DIR = ROOT / "examples"
IMAGES_DIR = EXAMPLES_DIR / "images"

ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_ZIP_NAME = "annotations/instances_val2017.json"  # nested inside the zip
ANN_VAL_NAME = "instances_val2017.json"
ANN_VAL_PATH = DATA_DIR / ANN_VAL_NAME
IMAGE_URL = "http://images.cocodataset.org/val2017/{id:012d}.jpg"


def _download(url: str, dest: Path) -> None:
    """Download with a simple progress indicator."""
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "visionforge-sample"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {pct}% ({done >> 20} MB / {total >> 20} MB)", end="", flush=True)
    print()


def ensure_val_annotations() -> dict:
    """Return the COCO val2017 annotation dict, downloading once if needed."""
    if ANN_VAL_PATH.exists():
        print(f"using cached {ANN_VAL_PATH}")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = DATA_DIR / "annotations_trainval2017.zip"
        _download(ANN_ZIP_URL, zip_path)
        print(f"extracting {ANN_VAL_NAME}...")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(ANN_ZIP_NAME) as src, open(ANN_VAL_PATH, "wb") as dst:
                dst.write(src.read())
        zip_path.unlink()  # free disk, keep only the extracted JSON
    with open(ANN_VAL_PATH) as f:
        return json.load(f)


def pick_images(anns: dict, count: int, seed: int) -> list[int]:
    """Deterministically pick images with several annotations across >= 2 categories."""
    # annotations per image
    anns_by_img: dict[int, list] = {}
    for ann in anns["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    candidates = []
    for img in anns["images"]:
        img_anns = anns_by_img.get(img["id"], [])
        cats = {a["category_id"] for a in img_anns}
        if len(img_anns) >= 3 and len(cats) >= 2:
            candidates.append(img["id"])

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def build_filtered_gt(anns: dict, image_ids: list[int]) -> dict:
    """Subset the full annotation dict to the chosen images + their categories."""
    id_set = set(image_ids)
    keep_cat_ids = {a["category_id"] for a in anns["annotations"] if a["image_id"] in id_set}

    return {
        "images": [img for img in anns["images"] if img["id"] in id_set],
        "annotations": [a for a in anns["annotations"] if a["image_id"] in id_set],
        "categories": [c for c in anns["categories"] if c["id"] in keep_cat_ids],
    }


def download_images(image_ids: list[int], max_size: int) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for img_id in image_ids:
        dest = IMAGES_DIR / f"{img_id:012d}.jpg"
        if dest.exists():
            print(f"  {dest.name} exists, skipping")
            continue
        url = IMAGE_URL.format(id=img_id)
        req = urllib.request.Request(url, headers={"User-Agent": "visionforge-sample"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # downscale to keep the repo small
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        img.save(dest, "JPEG", quality=85)
        print(f"  saved {dest.name} ({dest.stat().st_size >> 10} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small COCO val sample for VisionForge demo.")
    parser.add_argument("--count", type=int, default=12, help="number of images to include")
    parser.add_argument("--seed", type=int, default=42, help="random seed for image selection")
    parser.add_argument("--max-size", type=int, default=640, help="max image dimension after downscale")
    args = parser.parse_args()

    anns = ensure_val_annotations()
    image_ids = pick_images(anns, args.count, args.seed)
    print(f"picked {len(image_ids)} images: {image_ids}")

    download_images(image_ids, args.max_size)

    gt = build_filtered_gt(anns, image_ids)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    gt_path = EXAMPLES_DIR / "gt_sample.json"
    gt_path.write_text(json.dumps(gt, indent=2))
    print(f"wrote {gt_path}")

    # sanity summary
    total_anns = len(gt["annotations"])
    cats = {c["name"] for c in gt["categories"]}
    print(f"GT summary: {len(image_ids)} images, {total_anns} annotations, {len(cats)} categories: {sorted(cats)}")


if __name__ == "__main__":
    main()
