import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

# YOLO from ultralytics
from ultralytics import YOLO

# Official RT-DETR from transformers
from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor

# COCO evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import warnings
warnings.filterwarnings(
    "ignore",
    message="Could not load the custom kernel for multi-scale deformable attention"
)

warnings.filterwarnings(
    "ignore",
    message="The tied weights mapping and config for this model specifies to tie"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Generate runtime identifier
RUNTIME_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n{'='*80}")
print(f"SMALL OBJECT DETECTION BENCHMARK - FP16 - RUNTIME ID: {RUNTIME_ID}")
print(f"{'='*80}\n")

# Paths
BASE_DIR = "/p/composml/object-detection-benchmark"

# COCO
COCO_DIR = os.path.join(BASE_DIR, "coco_2017")
COCO_IMGS = os.path.join(COCO_DIR, "images/val2017")
COCO_ANN = os.path.join(COCO_DIR, "annotations/instances_val2017.json")

# Open Images
OI_BASE_VAL = os.path.join(BASE_DIR, "datasets/open-images-v7/open-images-v7/validation")
OI_IMGS = os.path.join(OI_BASE_VAL, "data")
OI_LABELS = os.path.join(OI_BASE_VAL, "labels/detections.csv")
OI_CLASSES = os.path.join(OI_BASE_VAL, "metadata/classes.csv")

# Ultralytics
ULTRALYTICS_DIR = os.path.join(BASE_DIR, "ultralytics/ultralytics")
COCO_YAML = os.path.join(ULTRALYTICS_DIR, "cfg/datasets/coco.yaml")

# Experiment output
TODAY = datetime.now().strftime("%Y-%m-%d")

EXP_ROOT = "small_object_benchmark_fp16"
OUT_DIR = os.path.join(EXP_ROOT, TODAY)
os.makedirs(OUT_DIR, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Fixed batch sizes
BATCH_SIZE = 16
NUM_WORKERS = 4

# The 24 matched classes between COCO and Open Images
MATCHED_CLASSES = {
    'apple', 'backpack', 'baseball glove', 'bird', 'boat', 'book',
    'bottle', 'car', 'clock', 'handbag', 'kite', 'knife',
    'mouse',  # COCO class (computer mouse)
    'orange', 'parking meter', 'person', 'sheep', 'snowboard',
    'spoon', 'tie', 'toothbrush', 'traffic light', 'vase', 'wine glass'
}

# Explicit mapping for OpenImages
OPENIMAGES_CLASS_MAPPING = {
    "mouse": "Computer mouse"  # Force alignment
}

# ============================================================================
# MODEL DEFINITIONS - WITH OFFICIAL RT-DETR
# ============================================================================

MODELS_TO_EVAL = [
    # YOLOv5u series (Ultralytics - FP16)
    ("yolov5nu.pt", "yolov5nu", "YOLOv5u", "#9B59B6", "ultralytics"),
    ("yolov5su.pt", "yolov5su", "YOLOv5u", "#9B59B6", "ultralytics"),
    ("yolov5mu.pt", "yolov5mu", "YOLOv5u", "#9B59B6", "ultralytics"),
    ("yolov5lu.pt", "yolov5lu", "YOLOv5u", "#9B59B6", "ultralytics"),
    ("yolov5xu.pt", "yolov5xu", "YOLOv5u", "#9B59B6", "ultralytics"),
    
    # YOLO11 series (Ultralytics - FP16)
    ("yolo11n.pt", "yolo11n", "YOLO11", "#1E3A8A", "ultralytics"),
    ("yolo11s.pt", "yolo11s", "YOLO11", "#1E3A8A", "ultralytics"),
    ("yolo11m.pt", "yolo11m", "YOLO11", "#1E3A8A", "ultralytics"),
    ("yolo11l.pt", "yolo11l", "YOLO11", "#1E3A8A", "ultralytics"),
    ("yolo11x.pt", "yolo11x", "YOLO11", "#1E3A8A", "ultralytics"),

    # RT-DETR series (Ultralytics - FP16 NATIVE SUPPORT!)
    ("rtdetr-l.pt", "rtdetr-l", "RT-DETR-Ultralytics", "#92D050", "ultralytics"),
    ("rtdetr-x.pt", "rtdetr-x", "RT-DETR-Ultralytics", "#92D050", "ultralytics"),
    
    # # RT-DETR series (Official HuggingFace - FP16 supported!)
    # ("PekingU/rtdetr_r50vd",   "rtdetr-r50",   "RT-DETR-HF",  "#8B5F70", "transformers"),
    # ("PekingU/rtdetr_r101vd",  "rtdetr-r101",  "RT-DETR-HF",  "#8B5F70", "transformers"),
    
    # # RT-DETRv2 series (Official HuggingFace - FP16 supported!)
    # ("PekingU/rtdetr_v2_r50vd",  "rtdetrv2-r50", "RT-DETRv2-HF", "#DAA520", "transformers"),
    # ("PekingU/rtdetr_v2_r101vd", "rtdetrv2-r101", "RT-DETRv2-HF", "#DAA520", "transformers"),
]

# ============================================================================
# DATASET CLASSES
# ============================================================================

class CocoSmallObjectDataset(Dataset):
    def __init__(self, img_dir, ann_file, matched_classes, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.coco = COCO(ann_file)

        self.matched_cat_ids = []
        self.cat_id_to_name = {}
        for cat_id, cat_info in self.coco.cats.items():
            if cat_info['name'] in matched_classes:
                self.matched_cat_ids.append(cat_id)
                self.cat_id_to_name[cat_id] = cat_info['name']

        self.img_ids = []
        for cat_id in self.matched_cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        self.img_ids = sorted(list(set(self.img_ids)))
        
        self.class_name_to_id = {v: k for k, v in self.cat_id_to_name.items()}
        print(f"  → COCO: {len(self.img_ids)} images with matched classes")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (width, height)
        if self.transform:
            img_tensor = self.transform(img)
        return img_tensor, img_id, (orig_size[0], orig_size[1])


class OpenImagesSmallObjectDataset(Dataset):
    def __init__(self, img_dir, labels_csv, classes_csv, matched_classes, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        classes_df = pd.read_csv(classes_csv, header=None, names=["LabelName", "DisplayName"])
        self.class_map = dict(zip(classes_df["DisplayName"].str.lower(), classes_df["LabelName"]))

        self.matched_mids = {}
        for cls in matched_classes:
            cls_lower = cls.lower()
            oi_name = OPENIMAGES_CLASS_MAPPING.get(cls_lower, cls_lower).lower()
            if oi_name in self.class_map:
                self.matched_mids[self.class_map[oi_name]] = cls

        sorted_classes = sorted(matched_classes)
        self.class_name_to_cat_id = {name: idx + 1 for idx, name in enumerate(sorted_classes)}
        self.cat_id_to_class_name = {idx + 1: name for idx, name in enumerate(sorted_classes)}

        print("  → Loading Open Images detections...")
        dets_df = pd.read_csv(labels_csv)
        dets_df = dets_df[dets_df["Confidence"] == 1]
        dets_df = dets_df[dets_df["LabelName"].isin(self.matched_mids.keys())]

        candidate_img_ids = dets_df["ImageID"].unique().tolist()
        print("  → Checking which images exist on disk...")
        self.img_ids = []
        for img_id in tqdm(candidate_img_ids, desc="  Verifying images"):
            if os.path.exists(os.path.join(self.img_dir, f"{img_id}.jpg")):
                self.img_ids.append(img_id)
        self.img_ids = sorted(self.img_ids)
        print(f"  → Open Images: {len(self.img_ids)} images with matched classes (verified)")
        self.img_id_to_idx = {img_id: idx + 1 for idx, img_id in enumerate(self.img_ids)}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Missing image → black placeholder: {img_path}")
            img = Image.new("RGB", (640, 640), (0, 0, 0))
        orig_size = img.size  # (width, height)
        if self.transform:
            img_tensor = self.transform(img)
        return img_tensor, img_id, (orig_size[0], orig_size[1])



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_exp_dir(model_name, dataset):
    """Create experiment directory under today's date folder"""
    path = os.path.join(OUT_DIR, f"{RUNTIME_ID}_{dataset}_{model_name}")
    os.makedirs(path, exist_ok=True)
    return path

def count_params(model, model_type="ultralytics"):
    """Count model parameters in millions"""
    if model_type == "ultralytics":
        return sum(p.numel() for p in model.model.parameters()) / 1e6
    else:  # transformers
        return sum(p.numel() for p in model.parameters()) / 1e6


def measure_fps_yolo(model, loader, warmup_batches=80, timing_batches=150):
    """Measure FPS for YOLO models with FP16"""
    model.model.eval()
    
    # Warmup
    print("    Warming up...")
    with torch.inference_mode():
        for i, (img_batch, _, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            _ = model(img_batch.to(DEVICE), verbose=False, half=True, imgsz=640)
    torch.cuda.synchronize()
    time.sleep(0.5)

    
    # Timing
    print("    Measuring inference speed...")
    times = []
    with torch.inference_mode():
        for i, (img_batch, _, _) in enumerate(loader):
            if i >= timing_batches:
                break
            bs = img_batch.shape[0]
            if bs < BATCH_SIZE:
                continue
            torch.cuda.synchronize()
            start = time.time()
            _ = model(img_batch.to(DEVICE), verbose=False, half=True, imgsz=640)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.extend([elapsed / bs] * bs)

    avg_time = np.median(times)
    return round(1 / avg_time if avg_time > 0 else 0.0, 2)



def measure_fps_rtdetr(model, processor, loader, warmup_batches=80, timing_batches=150):
    """Measure FPS for RT-DETR models with AMP"""
    model.eval()

    # Warmup
    print("    Warming up...")
    with torch.inference_mode():
        for i, (img_batch, _, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            # img_batch is a tensor [B,3,640,640]; processor handles tensors fine
            inputs = processor(images=img_batch, return_tensors="pt").to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = processor.post_process_object_detection(
                    model(**inputs), 
                    target_sizes=torch.tensor([[640, 640]] * img_batch.shape[0]), 
                    threshold=0.001
                )
    torch.cuda.synchronize()
    time.sleep(0.5)


    # Timing
    print("    Measuring inference speed...")
    times = []
    with torch.inference_mode():
        for i, (img_batch, _, _) in enumerate(loader):
            if i >= timing_batches:
                break
            bs = img_batch.shape[0]
            if bs < BATCH_SIZE:
                continue
            torch.cuda.synchronize()
            start = time.time()
            inputs = processor(images=img_batch, return_tensors="pt").to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = processor.post_process_object_detection(
                    model(**inputs), 
                    target_sizes=torch.tensor([[640, 640]] * img_batch.shape[0]), 
                    threshold=0.001
                )
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.extend([elapsed / bs] * bs)

    avg_time = np.median(times)
    return round(1 / avg_time if avg_time > 0 else 0.0, 2)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

    
def evaluate_coco_yolo(model, exp_dir):
    """
    Use Ultralytics built-in val() for COCO — correct AP + fast speed measurement.
    """
    print("    Running COCO validation (model.val)...")
    
    metrics = model.val(
        data=COCO_YAML,
        imgsz=640,
        batch=BATCH_SIZE,
        half=True,
        project=exp_dir,
        name="val",
        save_json=True,
        verbose=False
    )

    # Read directly from model.val() — correct standard COCO eval over all 80 classes
    ap_all   = metrics.box.map        # mAP50-95 all areas
    ap_small = 0.0
    ap50_small = 0.0
    pred_path = os.path.join(exp_dir, "val", "predictions.json")
    if os.path.exists(pred_path):
        coco_gt = COCO(COCO_ANN)
        coco_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
        coco_name_to_id = {v: k for k, v in coco_id_to_name.items()}
        matched_cat_ids = [coco_name_to_id[cls] for cls in MATCHED_CLASSES if cls in coco_name_to_id]
        matched_img_ids = sorted(set(
            img_id
            for cat_id in matched_cat_ids
            for img_id in coco_gt.getImgIds(catIds=[cat_id])
        ))

        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds = matched_cat_ids
        coco_eval.params.imgIds = matched_img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_small   = coco_eval.stats[3]
        ap50_small = coco_eval.stats[6] if len(coco_eval.stats) > 6 else 0.0

    # ── return metrics too so main() can read speed ──
    return ap_all, ap_small, ap50_small, metrics




def evaluate_openimages_yolo(model, loader, dataset, gt_ann_path, exp_dir):
    """
    Evaluate YOLO models on filtered Open Images validation set
    Uses COCO-style class names for filtering → custom category IDs from dataset
    """
    print("    Collecting predictions on Open Images (YOLO)...")
    model.model.eval()
    predictions = []

    coco_anno = COCO(COCO_ANN)
    coco_id_to_name = {cat['id']: cat['name'] for cat in coco_anno.loadCats(coco_anno.getCatIds())}

    cat_ids_sorted = sorted(coco_anno.getCatIds())
    yolo_idx_to_coco_id = {i: cat_ids_sorted[i] for i in range(len(cat_ids_sorted))}
        
    with torch.inference_mode():
        for img_batch, img_id_batch, orig_sizes in tqdm(loader, desc="    Inference"):
            results = model(img_batch.to(DEVICE), verbose=False, half=True, imgsz=640)

            for b in range(img_batch.shape[0]):
                img_id = img_id_batch[b]
                coco_img_id = dataset.img_id_to_idx[img_id]
                orig_w, orig_h = orig_sizes[b]
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0

                if results[b].boxes is None or len(results[b].boxes) == 0:
                    continue

                for box, conf, cls_idx in zip(
                    results[b].boxes.xyxy,
                    results[b].boxes.conf,
                    results[b].boxes.cls
                ):
                    cls_idx_int = int(cls_idx)
                    if cls_idx_int not in yolo_idx_to_coco_id:
                        continue
                    coco_cat_id = yolo_idx_to_coco_id[cls_idx_int]
                    class_name = coco_id_to_name[coco_cat_id]
                    if class_name not in dataset.class_name_to_cat_id:
                        continue

                    x1, y1, x2, y2 = box.cpu().numpy()
                    x1 *= scale_x; y1 *= scale_y
                    x2 *= scale_x; y2 *= scale_y

                    predictions.append({
                        "image_id": int(coco_img_id),
                        "category_id": int(dataset.class_name_to_cat_id[class_name]),
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "score": float(conf)
                    })

    if not predictions:
        print("    Warning: No predictions collected!")
        return 0.0, 0.0, 0.0

    print(f"    Collected {len(predictions)} predictions")
    pred_path = os.path.join(exp_dir, "openimages_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)

    coco_gt = COCO(gt_ann_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0], coco_eval.stats[3], coco_eval.stats[6] if len(coco_eval.stats) > 6 else 0.0



def evaluate_coco_rtdetr(model, processor, loader, exp_dir):
    print(" Collecting predictions on COCO (RT-DETR)...")
    model.eval()
    predictions = []

    coco_gt = COCO(COCO_ANN)
    coco_id_to_name = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    coco_name_to_id = {v: k for k, v in coco_id_to_name.items()}

    matched_cat_ids = [coco_name_to_id[cls] for cls in MATCHED_CLASSES if cls in coco_name_to_id]

    with torch.inference_mode():
        for img_batch, img_id_batch, orig_sizes in tqdm(loader, desc="    Inference"):
            inputs = processor(images=img_batch, return_tensors="pt").to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)

            target_sizes = torch.tensor([[h, w] for w, h in orig_sizes], device="cpu")

            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.001
            )

            for b in range(len(orig_sizes)):
                img_id = img_id_batch[b]

                for score, label, box in zip(
                    results[b]["scores"], results[b]["labels"], results[b]["boxes"]
                ):
                    cls_id = int(label.item())
                    if cls_id not in coco_id_to_name:
                        continue
                    class_name = coco_id_to_name[cls_id]
                    if class_name not in MATCHED_CLASSES:
                        continue

                    x1, y1, x2, y2 = box.cpu().numpy()
                    predictions.append({
                        "image_id": int(img_id),
                        "category_id": int(coco_name_to_id[class_name]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })

    if not predictions:
        print("    Warning: No predictions collected!")
        return 0.0, 0.0, 0.0

    print(f"    Collected {len(predictions)} predictions")
    pred_path = os.path.join(exp_dir, "coco_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = matched_cat_ids
    matched_img_ids = sorted(set(
    img_id
    for cat_id in matched_cat_ids
    for img_id in coco_gt.getImgIds(catIds=[cat_id])
    ))
    coco_eval.params.imgIds = matched_img_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0], coco_eval.stats[3], coco_eval.stats[6] if len(coco_eval.stats) > 6 else 0.0



def evaluate_openimages_rtdetr(model, processor, loader, dataset, gt_ann_path, exp_dir):
    print(" Collecting predictions on Open Images (RT-DETR)...")
    model.eval()
    predictions = []

    coco_anno = COCO(COCO_ANN)
    coco_id_to_name = {cat['id']: cat['name'] for cat in coco_anno.loadCats(coco_anno.getCatIds())}

    with torch.inference_mode():
        for img_batch, img_id_batch, orig_sizes in tqdm(loader, desc="    Inference"):
            inputs = processor(images=img_batch, return_tensors="pt").to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)

            target_sizes = torch.tensor(
                [[h, w] for w, h in orig_sizes],  # (H, W)
                device="cpu"
            )
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.001
            )

            for b in range(len(orig_sizes)):
                img_id = img_id_batch[b]
                coco_img_id = dataset.img_id_to_idx[img_id]

                for score, label, box in zip(
                    results[b]["scores"], results[b]["labels"], results[b]["boxes"]
                ):
                    cls_id = int(label.item())
                    if cls_id not in coco_id_to_name:
                        continue
                    class_name = coco_id_to_name[cls_id]
                    if class_name not in dataset.class_name_to_cat_id:
                        continue

                    x1, y1, x2, y2 = box.cpu().numpy()
                    predictions.append({
                        "image_id": int(coco_img_id),
                        "category_id": int(dataset.class_name_to_cat_id[class_name]),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })

    if not predictions:
        print("    Warning: No predictions collected!")
        return 0.0, 0.0, 0.0

    print(f"    Collected {len(predictions)} predictions")
    pred_path = os.path.join(exp_dir, "openimages_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)

    coco_gt = COCO(gt_ann_path)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0], coco_eval.stats[3], coco_eval.stats[6] if len(coco_eval.stats) > 6 else 0.0






def create_openimages_annotations(dataset, labels_csv, classes_csv, matched_classes, output_path):
    """Create COCO-format annotations for Open Images validation"""
    img_ids = dataset.img_ids
    img_id_to_idx = dataset.img_id_to_idx
    
    # Load class mapping
    classes_df = pd.read_csv(classes_csv, header=None, names=["LabelName", "DisplayName"])
    class_map = dict(zip(classes_df["DisplayName"].str.lower(), classes_df["LabelName"]))
    
    # Get MIDs for matched classes
    matched_mids = {}
    mid_to_catid = {}
    for i, cls in enumerate(sorted(matched_classes), 1):
        cls_lower = cls.lower()
        oi_name = OPENIMAGES_CLASS_MAPPING.get(cls_lower, cls_lower).lower()

        # Apply mouse disambiguation here too
        if oi_name in class_map:
            mid = class_map[oi_name]
            matched_mids[mid] = cls
            mid_to_catid[mid] = i
    
    # Load detections
    dets_df = pd.read_csv(labels_csv)
    dets_df = dets_df[dets_df["Confidence"] == 1]
    dets_df = dets_df[dets_df["ImageID"].isin(img_ids)]
    dets_df = dets_df[dets_df["LabelName"].isin(matched_mids.keys())]




    
    # Load image sizes
    print("    Loading image sizes...")
    img_sizes = {}
    for img_id in tqdm(img_ids, desc="    Image sizes"):
        try:
            with Image.open(os.path.join(OI_IMGS, f"{img_id}.jpg")) as img:
                img_sizes[img_id] = img.size
        except:
            pass
    
    # Create COCO format
    images, annotations = [], []
    ann_id = 1
    
    for img_id in img_ids:
        if img_id not in img_sizes:
            continue

        idx = img_id_to_idx[img_id]
        w, h = img_sizes[img_id]
        images.append({"id": idx, "file_name": f"{img_id}.jpg", "width": w, "height": h})

        for _, row in dets_df[dets_df["ImageID"] == img_id].iterrows():
            mid = row["LabelName"]
            if mid not in mid_to_catid:
                continue
            x1 = row["XMin"] * w
            y1 = row["YMin"] * h
            bbox_w = (row["XMax"] - row["XMin"]) * w
            bbox_h = (row["YMax"] - row["YMin"]) * h
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": mid_to_catid[mid],
                "bbox": [x1, y1, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            })
            ann_id += 1

    categories = [
        {"id": mid_to_catid[mid], "name": cls, "supercategory": "object"}
        for mid, cls in matched_mids.items()
    ]

    with open(output_path, 'w') as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    print(f"    → Created {len(images)} images, {len(annotations)} annotations")
    return output_path


# ============================================================================
# COLLATE
# ============================================================================

def custom_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    img_ids = [item[1] for item in batch]
    orig_sizes = [item[2] for item in batch]  # list of (W, H) tuples
    return imgs, img_ids, orig_sizes


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Run complete benchmark with FP16 for ALL models"""
    
    # Prepare datasets
    print("\n" + "="*80)
    print("PREPARING DATASETS")
    print("="*80)
    
    transform = transforms.Compose([
    transforms.Resize((640, 640)),   
    transforms.ToTensor(),
])
    
    print("\nLoading COCO dataset...")
    coco_dataset = CocoSmallObjectDataset(COCO_IMGS, COCO_ANN, MATCHED_CLASSES, transform)
    coco_loader = DataLoader(
        coco_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn
    )

    
    print("\nLoading Open Images dataset...")
    oi_dataset = OpenImagesSmallObjectDataset(OI_IMGS, OI_LABELS, OI_CLASSES, MATCHED_CLASSES, transform)
    oi_loader = DataLoader(
        oi_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn
    )

    
    print("\nCreating Open Images ground truth...")
    oi_gt_path = os.path.join(OUT_DIR, f"{RUNTIME_ID}_openimages_gt.json")
    create_openimages_annotations(oi_dataset, OI_LABELS, OI_CLASSES, MATCHED_CLASSES, oi_gt_path)

    results = {
        ds: {fam: {"latency": [], "ap_all": [], "ap_small": [], "ap50_small": [],
                   "params": [], "labels": [], "color": None}
             for fam in ["YOLOv5u", "YOLO11", "RT-DETR-Ultralytics", "RT-DETR-HF", "RT-DETRv2"]}
        for ds in ["coco", "openimages"]
    }

    
    # Evaluate each model
    print("\n" + "="*80)
    print("EVALUATING MODELS (FP16 + AMP)")
    print("="*80)
    
    for model_path, label, family, color, impl_type in MODELS_TO_EVAL:
        print(f"\n{'='*80}")
        print(f"Model: {label} ({family}) — {impl_type}")
        print(f"{'='*80}")

        try:
            if impl_type == "ultralytics":
                print("  Loading model...")
                model = YOLO(model_path).to(DEVICE)
                params = count_params(model, "ultralytics")

                # ── COCO ──
                print("\n  COCO Evaluation (FP16):")
                exp_dir_coco = create_exp_dir(label, "coco")

                fps_coco = measure_fps_yolo(model, coco_loader)
                latency_coco = 1000 / fps_coco if fps_coco > 0 else 0
                print(f"  FPS: {fps_coco:.2f} | Latency: {latency_coco:.2f} ms/img")
                
                ap_all_c, ap_small_c, ap50_small_c, metrics = evaluate_coco_yolo(model, exp_dir_coco)

                
                print(f"  FPS: {fps_coco:.2f} | Latency: {latency_coco:.2f} ms/img")
                print(f"  AP (all):    {ap_all_c:.4f}")
                print(f"  ★ AP (small): {ap_small_c:.4f} ★")
                print(f"  AP50 (small): {ap50_small_c:.4f}")

                for k, v in zip(
                    ["latency","ap_all","ap_small","ap50_small","params","labels"],
                    [latency_coco, ap_all_c, ap_small_c, ap50_small_c, params, label]
                ):
                    if k == "labels":
                        results["coco"][family][k].append(v)
                    else:
                        results["coco"][family][k].append(v)
                results["coco"][family]["color"] = color

                # ── Open Images ──
                print("\n  Open Images Evaluation (FP16):")
                exp_dir_oi = create_exp_dir(label, "openimages")
                fps_oi = measure_fps_yolo(model, oi_loader)
                latency_oi = 1000 / fps_oi if fps_oi > 0 else 0
                print(f"  FPS: {fps_oi:.2f} | Latency: {latency_oi:.2f} ms/img")

                ap_all_oi, ap_small_oi, ap50_small_oi = evaluate_openimages_yolo(
                    model, oi_loader, oi_dataset, oi_gt_path, exp_dir_oi
                )
                print(f"  AP (all):    {ap_all_oi:.4f}")
                print(f"  ★ AP (small): {ap_small_oi:.4f} ★")
                print(f"  AP50 (small): {ap50_small_oi:.4f}")

                results["openimages"][family]["latency"].append(latency_oi)
                results["openimages"][family]["ap_all"].append(ap_all_oi)
                results["openimages"][family]["ap_small"].append(ap_small_oi)
                results["openimages"][family]["ap50_small"].append(ap50_small_oi)
                results["openimages"][family]["params"].append(params)
                results["openimages"][family]["labels"].append(label)
                results["openimages"][family]["color"] = color
                
                # Save metrics
                for ds_name, exp_dir, fps, lat, ap_a, ap_s, ap50_s in [
                    ("COCO",           exp_dir_coco, fps_coco, latency_coco, ap_all_c,  ap_small_c,  ap50_small_c),
                    ("Open Images V7", exp_dir_oi,   fps_oi,   latency_oi,   ap_all_oi, ap_small_oi, ap50_small_oi),
                ]:
                    with open(os.path.join(exp_dir, "metrics.json"), 'w') as f:
                        json.dump({
                            "model": label, "family": family, "params_M": params,
                            "fps": fps, "latency_ms": lat,
                            "ap_all": ap_a, "ap_small": ap_s, "ap50_small": ap50_s,
                            "dataset": ds_name, "precision": "FP16"
                        }, f, indent=2)

                del model
                torch.cuda.empty_cache()

            else:  # transformers
                print(f"  Loading {model_path} from HuggingFace...")
                processor = RTDetrImageProcessor.from_pretrained(
                    model_path, do_rescale=False, do_resize=True, do_normalize=True
                )
                model_cls = RTDetrV2ForObjectDetection if "v2" in model_path else RTDetrForObjectDetection
                kwargs = {"tie_word_embeddings": False} if "v2" in model_path else {}
                model = model_cls.from_pretrained(model_path, **kwargs).to(DEVICE)
                params = count_params(model, "transformers")

                # ── COCO ──
                print("\n  COCO Evaluation (AMP):")
                exp_dir_coco = create_exp_dir(label, "coco")
                fps_coco = measure_fps_rtdetr(model, processor, coco_loader)
                latency_coco = 1000 / fps_coco if fps_coco > 0 else 0
                print(f"  FPS: {fps_coco:.2f} | Latency: {latency_coco:.2f} ms/img")

                ap_all_c, ap_small_c, ap50_small_c = evaluate_coco_rtdetr(
                    model, processor, coco_loader, exp_dir_coco
                )
                print(f"  AP (all):    {ap_all_c:.4f}")
                print(f"  ★ AP (small): {ap_small_c:.4f} ★")
                print(f"  AP50 (small): {ap50_small_c:.4f}")

                results["coco"][family]["latency"].append(latency_coco)
                results["coco"][family]["ap_all"].append(ap_all_c)
                results["coco"][family]["ap_small"].append(ap_small_c)
                results["coco"][family]["ap50_small"].append(ap50_small_c)
                results["coco"][family]["params"].append(params)
                results["coco"][family]["labels"].append(label)
                results["coco"][family]["color"] = color

                # ── Open Images ──
                print("\n  Open Images Evaluation (AMP):")
                exp_dir_oi = create_exp_dir(label, "openimages")
                fps_oi = measure_fps_rtdetr(model, processor, oi_loader)
                latency_oi = 1000 / fps_oi if fps_oi > 0 else 0
                print(f"  FPS: {fps_oi:.2f} | Latency: {latency_oi:.2f} ms/img")

                ap_all_oi, ap_small_oi, ap50_small_oi = evaluate_openimages_rtdetr(
                    model, processor, oi_loader, oi_dataset, oi_gt_path, exp_dir_oi
                )
                print(f"  AP (all):    {ap_all_oi:.4f}")
                print(f"  ★ AP (small): {ap_small_oi:.4f} ★")
                print(f"  AP50 (small): {ap50_small_oi:.4f}")

                results["openimages"][family]["latency"].append(latency_oi)
                results["openimages"][family]["ap_all"].append(ap_all_oi)
                results["openimages"][family]["ap_small"].append(ap_small_oi)
                results["openimages"][family]["ap50_small"].append(ap50_small_oi)
                results["openimages"][family]["params"].append(params)
                results["openimages"][family]["labels"].append(label)
                results["openimages"][family]["color"] = color

                for ds_name, exp_dir, fps, lat, ap_a, ap_s, ap50_s in [
                    ("COCO",           exp_dir_coco, fps_coco, latency_coco, ap_all_c,  ap_small_c,  ap50_small_c),
                    ("Open Images V7", exp_dir_oi,   fps_oi,   latency_oi,   ap_all_oi, ap_small_oi, ap50_small_oi),
                ]:
                    with open(os.path.join(exp_dir, "metrics.json"), 'w') as f:
                        json.dump({
                            "model": label, "family": family, "params_M": params,
                            "fps": fps, "latency_ms": lat,
                            "ap_all": ap_a, "ap_small": ap_s, "ap50_small": ap50_s,
                            "dataset": ds_name, "precision": "AMP (Mixed FP16/FP32)"
                        }, f, indent=2)

                del model, processor
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    
    # ============================================================================
    # VISUALIZATIONS
    # ============================================================================
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_dir = os.path.join(OUT_DIR, f"{RUNTIME_ID}_plots")
    os.makedirs(plot_dir, exist_ok=True)

    families = ["YOLOv5u", "YOLO11", "RT-DETR-Ultralytics"]

    for dataset_name, dataset_results in results.items():
        # Combined plot
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='black')
        ax.set_facecolor('black')

        for fam, data in dataset_results.items():
            if not data["latency"] or data["color"] is None:
                continue
            ax.plot(data["latency"], data["ap_small"], '-o', color=data["color"],
                    label=fam, linewidth=2.5, markersize=9)
            for i, lbl in enumerate(data["labels"]):
                ax.text(data["latency"][i] * 1.02, data["ap_small"][i], lbl,
                        fontsize=9, ha='left', va='center', color='white')

        ax.set_xlabel("Latency (ms/image)", fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel("AP Small (< 32×32 px, IoU=0.50:0.95)", fontsize=12, fontweight='bold', color='white')
        ax.set_title(f"Small Object Detection — {dataset_name.upper()}\n24 Matched Classes | FP16",
                     fontsize=14, fontweight='bold', pad=20, color='white')
        ax.grid(True, alpha=0.3, color='gray')
        ax.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_combined_AP_small.png"),
                    dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()

        # Per-family plots
        for family in families:
            data = dataset_results.get(family, {})
            if not data["latency"]:
                continue
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
            ax.set_facecolor('black')
            ax.plot(data["latency"], data["ap_small"], '-o', color=data["color"],
                    label=family, linewidth=2.5, markersize=9)
            for i, lbl in enumerate(data["labels"]):
                ax.text(data["latency"][i] * 1.02, data["ap_small"][i], lbl,
                        fontsize=9, ha='left', va='center', color='white')
            ax.set_xlabel("Latency (ms/image)", fontsize=12, fontweight='bold', color='white')
            ax.set_ylabel("AP Small (< 32×32 px)", fontsize=12, fontweight='bold', color='white')
            ax.set_title(f"{family} — {dataset_name.upper()}\n24 Classes | FP16",
                         fontsize=14, fontweight='bold', pad=20, color='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.legend(fontsize=11, facecolor='black', edgecolor='white', labelcolor='white')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{family}_AP_small.png"),
                        dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()

    
    # ============================================================================
    # TEXT REPORT
    # ============================================================================
    summary_path = os.path.join(OUT_DIR, f"{RUNTIME_ID}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    report_path = os.path.join(OUT_DIR, f"{RUNTIME_ID}_summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SMALL OBJECT DETECTION BENCHMARK — FP16 — SUMMARY REPORT\n")
        f.write(f"Runtime ID: {RUNTIME_ID}\n")
        f.write("="*80 + "\n\n")
        f.write("★ KEY METRIC: AP (small) — Objects < 32×32 pixels ★\n\n")

        for ds_name, ds_results in results.items():
            f.write(f"\n{'='*80}\nDATASET: {ds_name.upper()}\n{'='*80}\n\n")
            for family, data in ds_results.items():
                if not data["latency"]:
                    continue
                prec = "FP16" if family in ["YOLOv5u", "YOLO11", "RT-DETR-Ultralytics"] else "AMP"
                f.write(f"\n{family} ({prec}):\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Model':<18} {'Params(M)':<12} {'Latency(ms)':<14} {'★AP(small)★':<14} {'AP(all)':<10} {'AP50(sm)'}\n")
                f.write("-"*80 + "\n")
                for i, lbl in enumerate(data["labels"]):
                    f.write(f"{lbl:<18} {data['params'][i]:<12.2f} {data['latency'][i]:<14.2f} "
                            f"{data['ap_small'][i]:<14.6f} {data['ap_all'][i]:<10.4f} {data['ap50_small'][i]:.4f}\n")

        f.write("\n" + "="*80 + "\nCROSS-DATASET COMPARISON\n" + "="*80 + "\n\n")
        f.write(f"{'Family':<22}{'Model':<16}{'Δ AP(small)':>14}{'Rel.Drop%':>12}{'LatΔ%':>10}  Notes\n")
        f.write("-"*90 + "\n")
        for family in families:
            cd = results["coco"].get(family, {})
            od = results["openimages"].get(family, {})
            if not cd["labels"] or not od["labels"]:
                continue
            for i, lbl in enumerate(cd["labels"]):
                ap_c  = cd["ap_small"][i]
                ap_oi = od["ap_small"][i] if i < len(od["ap_small"]) else 0.0
                delta  = ap_c - ap_oi
                rel    = (delta / ap_c * 100) if ap_c > 0 else 0.0
                lat_c  = cd["latency"][i]
                lat_oi = od["latency"][i] if i < len(od["latency"]) else lat_c
                lat_d  = (lat_oi - lat_c) / lat_c * 100 if lat_c > 0 else 0.0
                note   = ("Very high drop" if rel > 70 else "High drop" if rel > 50 else
                          "Moderate drop" if rel > 30 else "")
                f.write(f"{family:<22}{lbl:<16}{delta:>14.4f}{rel:>11.1f}%{lat_d:>9.1f}%  {note}\n")

    print(f"\n✓ Outputs: {OUT_DIR}")
    print(f"✓ Plots:   {plot_dir}")
    print(f"✓ Report:  {report_path}")
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("★ KEY METRIC: AP (small) < 32×32 pixels ★")
    print("="*80)


if __name__ == "__main__":
    main()