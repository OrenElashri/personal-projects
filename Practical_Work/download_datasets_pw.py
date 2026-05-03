"""
Object Detection Benchmark Downloader
==============================================
Downloads the following datasets and model weights:

  Datasets:
    1. MS COCO 2017         (train / val / test)
    2. OpenImages V7        (train / validation / test)  — via FiftyOne

  Model Weights:
    3. Ultralytics          (YOLOv5u, YOLO11, RT-DETR)
    4. Hugging Face         (Official RT-DETR  + Official RT-DETRv2)
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_disk_space(directory: Path, required_gb: float) -> bool:
    try:
        _, _, free = shutil.disk_usage(directory)
        free_gb = free / (2 ** 30)
        print(f"   Available space : {free_gb:.1f} GB")
        print(f"   Required space  : ~{required_gb} GB")
        if free_gb < required_gb:
            print(f"\n⚠  Warning: only {free_gb:.1f} GB free — {required_gb} GB needed.")
            return input("   Continue anyway? (y/n): ").strip().lower() == "y"
        return True
    except Exception as e:
        print(f"⚠  Could not check disk space: {e}")
        return True

def run(cmd: list, env: dict | None = None) -> bool:
    """Run a subprocess and stream its output; return True on success."""
    print(f"→ {' '.join(str(c) for c in cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd,
            env=env or os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        return proc.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠  Interrupted — partially downloaded files will be resumed on next run.")
        proc.terminate()
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def pip_install(*packages: str) -> bool:
    return run([sys.executable, "-m", "pip", "install", "--quiet", *packages])

# ──────────────────────────────────────────────────────────────────────────────
# 1. MS COCO 2017
# ──────────────────────────────────────────────────────────────────────────────

COCO_FILES = {
    "train_images": (
        "http://images.cocodataset.org/zips/train2017.zip",
        "~18 GB",
    ),
    "val_images": (
        "http://images.cocodataset.org/zips/val2017.zip",
        "~1 GB",
    ),
    "test_images": (
        "http://images.cocodataset.org/zips/test2017.zip",
        "~6 GB",
    ),
    "annotations": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "~252 MB",
    ),
    "test_info": (
        "http://images.cocodataset.org/annotations/image_info_test2017.zip",
        "~1 MB",
    ),
}

COCO_SPLIT_SPACE = {
    "train": 18.5,
    "val": 1.0,
    "test": 6.5,
    "annotations": 0.3,
}

def _wget_download(url: str, dest_dir: Path) -> bool:
    """Download a file with wget (with resume support -c)."""
    try:
        import urllib.request, urllib.error
    except ImportError:
        pass

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_file = dest_dir / filename

    # Try wget first (best resume support)
    if shutil.which("wget"):
        return run(["wget", "-c", "--show-progress", "-P", str(dest_dir), url])

    # Fallback: curl
    if shutil.which("curl"):
        return run(["curl", "-C", "-", "-L", "-o", str(dest_file), url])

    # Pure Python fallback (no resume)
    print(f"⚠  wget/curl not found — using Python urllib (no resume support)")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest_file)
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def _unzip(zip_path: Path, dest_dir: Path) -> bool:
    if not zip_path.exists():
        print(f"⚠  {zip_path} not found, skipping unzip.")
        return False
    if shutil.which("unzip"):
        return run(["unzip", "-n", str(zip_path), "-d", str(dest_dir)])
    import zipfile
    print(f"→ Extracting {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

def download_coco(base_dir: Path, splits: list[str], keep_zips: bool = False):
    section("MS COCO 2017 — Download")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Map user selections to actual download keys
    key_map = {
        "train": ["train_images", "annotations"],
        "val":   ["val_images",   "annotations"],
        "test":  ["test_images",  "test_info"],
    }

    # Deduplicate (annotations may be needed by both train & val)
    keys_to_download: list[str] = []
    for split in splits:
        for k in key_map.get(split, []):
            if k not in keys_to_download:
                keys_to_download.append(k)

    required_gb = sum(COCO_SPLIT_SPACE.get(s, 0) for s in splits) + COCO_SPLIT_SPACE["annotations"]
    if not check_disk_space(base_dir, required_gb):
        return

    results = {}
    for key in keys_to_download:
        url, size = COCO_FILES[key]
        filename = url.split("/")[-1]
        zip_path = base_dir / filename
        print(f"\n▶ {key}  ({size})")

        if zip_path.exists():
            print(f"   ✓ Already downloaded: {zip_path.name}")
        else:
            ok = _wget_download(url, base_dir)
            results[key] = ok
            if not ok:
                print(f"   ✗ Failed to download {key}")
                continue

        # Unzip
        _unzip(zip_path, base_dir)

        if not keep_zips:
            zip_path.unlink(missing_ok=True)
            print(f"   🗑  Removed zip: {zip_path.name}")

    section("MS COCO 2017 — Done")
    print(f"Location: {base_dir}")
    print("""
Directory structure after download:
  coco2017/
    train2017/          ← training images
    val2017/            ← validation images
    test2017/           ← test images
    annotations/        ← JSON annotations
""")

# ──────────────────────────────────────────────────────────────────────────────
# 2. OpenImages V7  (via FiftyOne)
# ──────────────────────────────────────────────────────────────────────────────

OI_SPLITS_INFO = {
    "validation": {"images": "~41,000",      "size_gb": 5,   "size": "~5 GB"},
    "test":       {"images": "~125,000",     "size_gb": 36,  "size": "~36 GB"},
    "train":      {"images": "~1,743,042",   "size_gb": 510, "size": "~510 GB"},
}

def check_fiftyone() -> bool:
    try:
        import fiftyone as fo  # noqa: F401
        print(f"✓ FiftyOne {fo.__version__} detected")
        return True
    except ImportError:
        print("✗ FiftyOne not installed.  Installing …")
        return pip_install("fiftyone")

def download_openimages(base_dir: Path, splits: list[str], num_workers: int = 4):
    section("OpenImages V7 — Download (via FiftyOne)")

    if not check_fiftyone():
        print("✗ Could not install FiftyOne — skipping OpenImages download.")
        return

    base_dir.mkdir(parents=True, exist_ok=True)
    required_gb = sum(OI_SPLITS_INFO[s]["size_gb"] for s in splits)
    if not check_disk_space(base_dir, required_gb):
        return

    env = os.environ.copy()
    env["FIFTYONE_DATASET_ZOO_DIR"] = str(base_dir)

    for split in splits:
        info = OI_SPLITS_INFO[split]
        print(f"\n▶ {split}  ({info['images']} images, {info['size']})")
        cmd = [
            "fiftyone", "zoo", "datasets", "download", "open-images-v7",
            "--split", split,
            "--kwargs",
            "label_types=detections",
            "max_samples=-1",
            f"num_workers={num_workers}",
        ]
        ok = run(cmd, env=env)
        if ok:
            print(f"✓ {split} split downloaded.")
        else:
            print(f"✗ {split} split failed — rerun to resume.")

    section("OpenImages V7 — Done")
    print(f"Location: {base_dir}")
    print("""
Load in Python:
  import fiftyone as fo
  dataset = fo.load_dataset("open-images-v7-validation")
""")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Ultralytics Model Weights  (YOLOv5u, YOLO11, RT-DETR)
# ──────────────────────────────────────────────────────────────────────────────

ULTRALYTICS_MODELS = {
    # YOLOv5u  (re-trained with anchor-free Ultralytics head)
    "yolov5nu":      ("YOLOv5u Nano",   "~5 MB"),
    "yolov5su":      ("YOLOv5u Small",  "~15 MB"),
    "yolov5mu":      ("YOLOv5u Medium", "~41 MB"),
    "yolov5lu":      ("YOLOv5u Large",  "~88 MB"),
    "yolov5xu":      ("YOLOv5u XLarge", "~166 MB"),
    # YOLO11
    "yolo11n":       ("YOLO11 Nano",    "~5 MB"),
    "yolo11s":       ("YOLO11 Small",   "~19 MB"),
    "yolo11m":       ("YOLO11 Medium",  "~39 MB"),
    "yolo11l":       ("YOLO11 Large",   "~49 MB"),
    "yolo11x":       ("YOLO11 XLarge",  "~109 MB"),
    # RT-DETR (Ultralytics)
    "rtdetr-l":      ("RT-DETR Large",  "~66 MB"),
    "rtdetr-x":      ("RT-DETR XLarge", "~136 MB"),
}

ULTRALYTICS_GROUPS = {
    "yolov5u": ["yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu"],
    "yolo11":  ["yolo11n",  "yolo11s",  "yolo11m",  "yolo11l",  "yolo11x"],
    "rtdetr":  ["rtdetr-l", "rtdetr-x"],
}

def check_ultralytics() -> bool:
    try:
        import ultralytics  # noqa: F401
        print(f"✓ ultralytics {ultralytics.__version__} detected")
        return True
    except ImportError:
        print("✗ ultralytics not installed.  Installing …")
        return pip_install("ultralytics")

def download_ultralytics(dest_dir: Path, model_keys: list[str]):
    section("Ultralytics Model Weights — Download")

    if not check_ultralytics():
        print("✗ Could not install ultralytics — skipping.")
        return

    from ultralytics import YOLO  # noqa: E402 (import after install check)

    dest_dir.mkdir(parents=True, exist_ok=True)
    orig_dir = Path.cwd()
    os.chdir(dest_dir)  # ultralytics saves weights to cwd by default

    for key in model_keys:
        name, size = ULTRALYTICS_MODELS.get(key, (key, "unknown size"))
        pt_file = dest_dir / f"{key}.pt"
        if pt_file.exists():
            print(f"✓ Already present: {pt_file.name}  ({size})")
            continue
        print(f"\n▶ Downloading {name} ({size}) …")
        try:
            model = YOLO(f"{key}.pt")  # triggers auto-download
            # Move to dest_dir if it landed elsewhere
            local_pt = Path(f"{key}.pt")
            if local_pt.exists() and local_pt.resolve() != pt_file.resolve():
                shutil.move(str(local_pt), str(pt_file))
            print(f"✓ Saved: {pt_file}")
        except Exception as e:
            print(f"✗ Failed to download {key}: {e}")

    os.chdir(orig_dir)
    section("Ultralytics — Done")
    print(f"Location: {dest_dir}")
    print("""
Load in Python:
  from ultralytics import YOLO
  model = YOLO("yolov5nu.pt")
  model = YOLO("rtdetr-l.pt")
""")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Hugging Face  RT-DETR + RT-DETRv2
# ──────────────────────────────────────────────────────────────────────────────

HF_MODELS = {
    # Official RT-DETR (PekingU / Jozhang)
    "rtdetr-l-coco":        ("PekingU/rtdetr-l",          "RT-DETR Large (COCO)",    "~66 MB"),
    "rtdetr-x-coco":        ("PekingU/rtdetr-x",          "RT-DETR XLarge (COCO)",   "~136 MB"),
    "rtdetr-r50-coco":      ("jozhang97/rt-detr-r50",     "RT-DETR R50 (COCO)",      "~42 MB"),
    "rtdetr-r101-coco":     ("jozhang97/rt-detr-r101",    "RT-DETR R101 (COCO)",     "~76 MB"),
    # Official RT-DETRv2
    "rtdetrv2-r18-coco":    ("jozhang97/rt-detr-v2-r18",  "RT-DETRv2 R18 (COCO)",   "~33 MB"),
    "rtdetrv2-r50-coco":    ("jozhang97/rt-detr-v2-r50",  "RT-DETRv2 R50 (COCO)",   "~42 MB"),
    "rtdetrv2-r101-coco":   ("jozhang97/rt-detr-v2-r101", "RT-DETRv2 R101 (COCO)",  "~76 MB"),
}

HF_GROUPS = {
    "rtdetr":   ["rtdetr-l-coco", "rtdetr-x-coco", "rtdetr-r50-coco", "rtdetr-r101-coco"],
    "rtdetrv2": ["rtdetrv2-r18-coco", "rtdetrv2-r50-coco", "rtdetrv2-r101-coco"],
}

def check_transformers() -> bool:
    ok = True
    for pkg in ("transformers", "torch"):
        try:
            __import__(pkg)
            print(f"✓ {pkg} detected")
        except ImportError:
            print(f"✗ {pkg} not installed.  Installing …")
            ok = ok and pip_install(pkg)
    return ok

def download_hf_models(dest_dir: Path, model_keys: list[str]):
    section("Hugging Face RT-DETR / RT-DETRv2 — Download")

    if not check_transformers():
        print("✗ Required packages missing — skipping HF download.")
        return

    from transformers import AutoModelForObjectDetection, AutoImageProcessor  # noqa: E402

    dest_dir.mkdir(parents=True, exist_ok=True)

    for key in model_keys:
        repo_id, name, size = HF_MODELS.get(key, (key, key, "unknown"))
        save_path = dest_dir / key
        if (save_path / "config.json").exists():
            print(f"✓ Already present: {save_path.name}  ({size})")
            continue

        print(f"\n▶ Downloading {name} from {repo_id}  ({size}) …")
        try:
            processor = AutoImageProcessor.from_pretrained(repo_id)
            model = AutoModelForObjectDetection.from_pretrained(repo_id)
            save_path.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(str(save_path))
            model.save_pretrained(str(save_path))
            print(f"✓ Saved: {save_path}")
        except Exception as e:
            print(f"✗ Failed to download {key} ({repo_id}): {e}")

    section("Hugging Face — Done")
    print(f"Location: {dest_dir}")
    print("""
Load in Python:
  from transformers import AutoModelForObjectDetection, AutoImageProcessor
  processor = AutoImageProcessor.from_pretrained("path/to/rtdetr-l-coco")
  model     = AutoModelForObjectDetection.from_pretrained("path/to/rtdetr-l-coco")
""")

# ──────────────────────────────────────────────────────────────────────────────
# Interactive menus
# ──────────────────────────────────────────────────────────────────────────────

def ask_dir(prompt: str, default: str) -> Path:
    print(f"\n{prompt}")
    print(f"   Default: {default}")
    raw = input("→ Path (Enter for default): ").strip()
    p = Path(raw).expanduser().resolve() if raw else Path(default)
    p.mkdir(parents=True, exist_ok=True)
    print(f"✓ Using: {p}")
    return p

def menu_coco(base: Path):
    section("MS COCO 2017 — Split Selection")
    print("  1. train       (~18 GB images + annotations)")
    print("  2. val         (~1 GB  images + annotations)")
    print("  3. test        (~6 GB  images)")
    print("  4. essential   (val only)")
    print("  5. all         (train + val + test)")
    print("  q. skip COCO")
    choice = input("\nYour choice (e.g. '1 2', 'all', 'essential'): ").strip().lower()
    if choice in ("q", "skip"):
        return
    if choice == "all":
        splits = ["train", "val", "test"]
    elif choice == "essential":
        splits = ["val"]
    else:
        map_idx = {"1": "train", "2": "val", "3": "test"}
        splits = [map_idx[c] for c in choice.split() if c in map_idx]
    if not splits:
        print("No valid selection — skipping COCO.")
        return
    keep = input("Keep zip files after extraction? (y/n, default n): ").strip().lower() == "y"
    coco_dir = ask_dir("COCO 2017 download directory:", str(base / "coco2017"))
    download_coco(coco_dir, splits, keep_zips=keep)

def menu_openimages(base: Path):
    section("OpenImages V7 — Split Selection")
    for i, (split, info) in enumerate(OI_SPLITS_INFO.items(), 1):
        print(f"  {i}. {split:12s}  {info['images']:>12s} images  {info['size']}")
    print("  essential  → validation only")
    print("  all        → all splits  (WARNING: >500 GB)")
    print("  q          → skip")
    choice = input("\nYour choice: ").strip().lower()
    if choice in ("q", "skip"):
        return
    if choice == "all":
        splits = list(OI_SPLITS_INFO)
    elif choice == "essential":
        splits = ["validation"]
    else:
        map_idx = {"1": "validation", "2": "test", "3": "train"}
        splits = [map_idx[c] for c in choice.split() if c in map_idx]
    if not splits:
        print("No valid selection — skipping OpenImages.")
        return
    w = input("Number of download workers (default 4): ").strip()
    workers = int(w) if w.isdigit() else 4
    oi_dir = ask_dir("OpenImages V7 download directory:", str(base / "open-images-v7"))
    download_openimages(oi_dir, splits, num_workers=workers)

def menu_ultralytics(base: Path):
    section("Ultralytics Model Weights — Selection")
    all_keys: list[str] = []
    idx_map: dict[str, str] = {}
    counter = 1
    for group, keys in ULTRALYTICS_GROUPS.items():
        print(f"\n  [{group.upper()}]")
        for k in keys:
            name, size = ULTRALYTICS_MODELS[k]
            print(f"    {counter:2d}. {k:20s}  {name}  ({size})")
            idx_map[str(counter)] = k
            all_keys.append(k)
            counter += 1

    print("\n  Options: enter numbers (e.g. '1 6 13'), group name (e.g. 'yolov5u'),")
    print("           'all' for everything, or 'q' to skip")
    choice = input("\nYour choice: ").strip().lower()
    if choice in ("q", "skip"):
        return
    if choice == "all":
        selected = all_keys
    elif choice in ULTRALYTICS_GROUPS:
        selected = ULTRALYTICS_GROUPS[choice]
    else:
        selected = [idx_map[c] for c in choice.split() if c in idx_map]
    if not selected:
        print("No valid selection — skipping Ultralytics.")
        return
    ul_dir = ask_dir("Ultralytics weights directory:", str(base / "ultralytics_weights"))
    download_ultralytics(ul_dir, selected)

def menu_hf(base: Path):
    section("Hugging Face RT-DETR / RT-DETRv2 — Selection")
    all_keys: list[str] = []
    idx_map: dict[str, str] = {}
    counter = 1
    for group, keys in HF_GROUPS.items():
        print(f"\n  [{group.upper()}]")
        for k in keys:
            _, name, size = HF_MODELS[k]
            print(f"    {counter:2d}. {k:28s}  {name}  ({size})")
            idx_map[str(counter)] = k
            all_keys.append(k)
            counter += 1

    print("\n  Options: numbers, group name ('rtdetr' or 'rtdetrv2'), 'all', or 'q'")
    choice = input("\nYour choice: ").strip().lower()
    if choice in ("q", "skip"):
        return
    if choice == "all":
        selected = all_keys
    elif choice in HF_GROUPS:
        selected = HF_GROUPS[choice]
    else:
        selected = [idx_map[c] for c in choice.split() if c in idx_map]
    if not selected:
        print("No valid selection — skipping HF models.")
        return
    hf_dir = ask_dir("Hugging Face models directory:", str(base / "hf_models"))
    download_hf_models(hf_dir, selected)

# ──────────────────────────────────────────────────────────────────────────────
# CLI  (non-interactive)
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Object Detection Benchmark Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Interactive mode (default)
  python download_datasets.py

  # Download COCO val + OpenImages validation non-interactively
  python download_datasets.py --coco val --openimages validation \\
    --base-dir /data/benchmarks

  # Download all Ultralytics yolov5u + HF RT-DETR models
  python download_datasets.py --ultralytics yolov5u --hf rtdetr \\
    --base-dir /data/benchmarks
""",
    )
    parser.add_argument("--base-dir",      default="./benchmark_data",
                        help="Root directory for all downloads")
    parser.add_argument("--coco",          nargs="+",
                        choices=["train", "val", "test", "all"],
                        help="COCO 2017 splits to download")
    parser.add_argument("--openimages",    nargs="+",
                        choices=["train", "validation", "test", "all"],
                        help="OpenImages V7 splits to download")
    parser.add_argument("--ultralytics",   nargs="+",
                        help="Ultralytics model keys or group names (yolov5u, yolo11, rtdetr, all)")
    parser.add_argument("--hf",            nargs="+",
                        help="HuggingFace model keys or group names (rtdetr, rtdetrv2, all)")
    parser.add_argument("--oi-workers",    type=int, default=4,
                        help="OpenImages download workers (default: 4)")
    parser.add_argument("--keep-zips",     action="store_true",
                        help="Keep COCO zip files after extraction")
    return parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  OBJECT DETECTION BENCHMARK — UNIFIED DOWNLOADER")
    print("=" * 70)
    print("  Covers: MS COCO 2017 · OpenImages V7 · Ultralytics · HuggingFace")
    print("=" * 70)

    args = parse_args()
    base = Path(args.base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    non_interactive = any([args.coco, args.openimages, args.ultralytics, args.hf])

    if non_interactive:
        # ── Non-interactive ────────────────────────────────────────────────
        if args.coco:
            splits = list({"train", "val", "test"} & set(args.coco)) if "all" not in args.coco else ["train", "val", "test"]
            download_coco(base / "coco2017", splits, keep_zips=args.keep_zips)

        if args.openimages:
            splits = list(OI_SPLITS_INFO) if "all" in args.openimages else args.openimages
            download_openimages(base / "open-images-v7", splits, num_workers=args.oi_workers)

        if args.ultralytics:
            keys: list[str] = []
            for k in args.ultralytics:
                if k == "all":
                    keys = list(ULTRALYTICS_MODELS)
                    break
                keys.extend(ULTRALYTICS_GROUPS.get(k, [k]))
            download_ultralytics(base / "ultralytics_weights", keys)

        if args.hf:
            keys = []
            for k in args.hf:
                if k == "all":
                    keys = list(HF_MODELS)
                    break
                keys.extend(HF_GROUPS.get(k, [k]))
            download_hf_models(base / "hf_models", keys)

    else:
        # ── Interactive ───────────────────────────────────────────────────
        print(f"\nBase directory: {base}")
        print("\nThis script can download:")
        print("  1. MS COCO 2017        (images + annotations)")
        print("  2. OpenImages V7       (via FiftyOne)")
        print("  3. Ultralytics Weights (YOLOv5u/11, RT-DETR)")
        print("  4. HuggingFace Models  (Official RT-DETR, RT-DETRv2)")

        if input("\nDownload MS COCO 2017? (y/n): ").strip().lower() == "y":
            menu_coco(base)

        if input("\nDownload OpenImages V7? (y/n): ").strip().lower() == "y":
            menu_openimages(base)

        if input("\nDownload Ultralytics model weights? (y/n): ").strip().lower() == "y":
            menu_ultralytics(base)

        if input("\nDownload HuggingFace RT-DETR / RT-DETRv2 models? (y/n): ").strip().lower() == "y":
            menu_hf(base)

    section("All Done")
    print(f"All downloads saved under: {base}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠  Script interrupted — rerun to resume incomplete downloads.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)