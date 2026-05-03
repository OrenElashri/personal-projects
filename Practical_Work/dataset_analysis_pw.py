import json
import os
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Generate runtime identifier
RUNTIME_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n{'='*80}")
print(f"RUNTIME ID: {RUNTIME_ID}")
print(f"{'='*80}\n")

SMALL_THRESH = 32 * 32

BASE_DIR = "/home/elashrio/bachelor_thesis"

# Use today's date as folder name (2026-03-03 style)
TODAY = datetime.now().strftime("%Y-%m-%d")

COCO_OUT_DIR = os.path.join(BASE_DIR, "coco_analysis", TODAY)
OI_OUT_DIR   = os.path.join(BASE_DIR, "openimages_analysis", TODAY)

os.makedirs(COCO_OUT_DIR, exist_ok=True)
os.makedirs(OI_OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OI_OUT_DIR, "comparison"), exist_ok=True)



# COCO paths
COCO_ANN_TRAIN = "/p/composml/object-detection-benchmark/coco_2017/annotations/instances_train2017.json"
COCO_ANN_VAL = "/p/composml/object-detection-benchmark/coco_2017/annotations/instances_val2017.json"

# Open Images paths
OI_BASE_VAL = "/p/composml/object-detection-benchmark/datasets/open-images-v7/open-images-v7/validation"
OI_BASE_TRAIN = "/p/composml/object-detection-benchmark/datasets/open-images-v7/open-images-v7/train"
OI_LABELS_VAL = os.path.join(OI_BASE_VAL, "labels", "detections.csv")
OI_LABELS_TRAIN = os.path.join(OI_BASE_TRAIN, "labels", "detections.csv")
OI_CLASSES_CSV = os.path.join(OI_BASE_VAL, "metadata", "classes.csv")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_oi_image_sizes(data_dirs):
    img_sizes = {}
    all_jpg_files = []
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            jpg_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            all_jpg_files.extend([(data_dir, f) for f in jpg_files])
    
    print(f"Loading {len(all_jpg_files):,} image sizes...")
    
    for data_dir, filename in tqdm(all_jpg_files, desc="Image sizes", unit="img"):
        img_id = filename[:-4]
        try:
            with Image.open(os.path.join(data_dir, filename)) as img:
                img_sizes[img_id] = img.size
        except Exception:
            pass
    
    print(f"→ Loaded {len(img_sizes):,} image sizes")
    return img_sizes


def count_small_objects(annotations, area_threshold):
    per_class_total = defaultdict(int)
    per_class_small = defaultdict(int)
    
    for ann in tqdm(annotations, desc="Counting objects", unit="ann"):
        cid = ann["category_id"]
        w, h = ann["bbox"][2], ann["bbox"][3]
        area = w * h
        per_class_total[cid] += 1
        if area < area_threshold:
            per_class_small[cid] += 1
    
    return per_class_total, per_class_small


def count_small_objects_oi(detections, img_sizes, area_threshold, filter_classes=None):
    per_class_total = defaultdict(int)
    per_class_small = defaultdict(int)
    missing_size = 0

    # ===================================================================
    # IMPORTANT: RENAME "Computer mouse" → "mouse" FIRST (before filtering!)
    # ===================================================================
    detections = detections.copy()  # avoid modifying original
    detections['DisplayName'] = detections['DisplayName'].replace('Computer mouse', 'mouse')

    # Now apply filtering (using the renamed name "mouse")
    if filter_classes:
        filter_set = {fc.lower() for fc in filter_classes}
        detections = detections[detections['DisplayName'].str.lower().isin(filter_set)].copy()

    # Vectorized computation (rest unchanged)
    detections = detections.dropna(subset=['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'DisplayName'])

    detections['width'] = detections['ImageID'].map(lambda x: img_sizes.get(x, (0,0))[0])
    detections['height'] = detections['ImageID'].map(lambda x: img_sizes.get(x, (0,0))[1])

    valid = (detections['width'] > 0) & (detections['height'] > 0)
    missing_size = len(detections) - valid.sum()
    detections = detections[valid].copy()

    detections['bbox_w'] = (detections['XMax'] - detections['XMin']) * detections['width']
    detections['bbox_h'] = (detections['YMax'] - detections['YMin']) * detections['height']
    detections['area'] = detections['bbox_w'] * detections['bbox_h']

    grouped = detections.groupby('DisplayName').agg(
        total=('area', 'size'),
        small=('area', lambda x: (x < area_threshold).sum())
    ).reset_index()

    for _, row in grouped.iterrows():
        name = row['DisplayName']
        per_class_total[name] = int(row['total'])
        per_class_small[name] = int(row['small'])

    if missing_size > 0:
        print(f"Warning: {missing_size:,} boxes skipped (no image size)")

    return per_class_total, per_class_small


def create_results_table(per_class_total, per_class_small, class_names_dict):
    results = []
    for cid, name in class_names_dict.items():
        total = per_class_total[cid]
        small = per_class_small[cid]
        if total == 0:
            continue
        pct = (small / total * 100)
        results.append((name, total, small, pct))
    
    results.sort(key=lambda x: x[3], reverse=True)
    return results

def get_plot_filename(dataset, split, chart_type, runtime_id):
    # dataset: "coco" or "openimages"
    # split: "train", "val", "combined"
    # chart_type: "proportion", "log", "bubble"
    return f"{runtime_id}_{dataset}_{split}_small_objects_{chart_type}.png"

def plot_proportion_chart(results, title, output_path):
    if not results:
        print(f"Skipping proportion chart (no data): {output_path}")
        return
        
    class_names = [r[0] for r in results]
    totals = np.array([r[1] for r in results])
    smalls = np.array([r[2] for r in results])
    percentages = np.array([r[3] for r in results])
    
    normalized_smalls = smalls / totals
    normalized_large = 1.0 - normalized_smalls
    
    fig, ax = plt.subplots(figsize=(16, 10))
    x_pos = np.arange(len(class_names))
    bar_width = 0.78
    
    bars_small = ax.bar(x_pos, normalized_smalls, bar_width,
                        label='Small (<32²)', color='#A0C0C0', edgecolor='black', linewidth=0.5)
    ax.bar(x_pos, normalized_large, bar_width, bottom=normalized_smalls,
           label='Medium + Large (≥32²)', color='#CBDEDE', edgecolor='black', linewidth=0.5)
    
    # Add percentage labels
    for bar, pct in zip(bars_small, percentages):
        height = bar.get_height()
        if height > 0.03:
            ax.text(bar.get_x() + bar.get_width()/2., height/2., f'{pct:.0f}%',
                    ha='center', va='center', fontsize=7.5, fontweight='bold', color='black')
    
    class_labels = [f'{name}\n({int(small):,})' for name, small in zip(class_names, smalls)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels, rotation=90, ha='right', fontsize=9)
    ax.set_xlabel('Object Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion (100% = all instances)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{int(y*100)}%' for y in np.arange(0, 1.1, 0.1)])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_train_val_small_stacked(coco_results_dict, title, output_path):
    """
    Stacked bar: for each class, show how many small objects come from Train vs Val
    Uses combined top-30 classes order
    """
    # Get top-30 classes from combined
    combined_results = coco_results_dict["combined"][:30]
    class_names = [r[0] for r in combined_results]

    # Collect small counts from train and val
    train_small = {}
    val_small = {}
    for split in ["train", "val"]:
        for name, _, small, _ in coco_results_dict[split]:
            if name in class_names:
                if split == "train":
                    train_small[name] = small
                else:
                    val_small[name] = small

    # Prepare data (some classes might be missing in val → 0)
    train_vals = [train_small.get(name, 0) for name in class_names]
    val_vals = [val_small.get(name, 0) for name in class_names]

    fig, ax = plt.subplots(figsize=(18, 10))
    x_pos = np.arange(len(class_names))

    bars_train = ax.bar(x_pos, train_vals, 0.78,
                        label='Small in Train', color='#A0C0C0', edgecolor='black')
    bars_val = ax.bar(x_pos, val_vals, 0.78, bottom=train_vals,
                      label='Small in Val', color='#80A0A0', edgecolor='black')

    # Add total small labels
    for i, (t, v) in enumerate(zip(train_vals, val_vals)):
        total = t + v
        if total > 200:
            ax.text(i, total + 500, f'{total:,}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{name}\n({t+v:,})" for name, t, v in zip(class_names, train_vals, val_vals)],
                       rotation=90, ha='right', fontsize=9)
    ax.set_xlabel('Object Class (total small instances)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Small Objects', fontsize=13, fontweight='bold')
    ax.set_title(title + "\n(Stacked: Train vs Val contribution)", fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_yscale('log')  # log scale because person dominates
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='auto'))
    ax.grid(axis='y', which='major', alpha=0.4, linestyle='--')
    ax.grid(axis='y', which='minor', alpha=0.15, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Train/Val small stacked plot: {output_path}")


def plot_log_chart(results, title, output_path, ylim_max=None):
    if not results:
        print(f"Skipping log chart (no data): {output_path}")
        return
        
    class_names = [r[0] for r in results]
    totals = np.array([r[1] for r in results])
    smalls = np.array([r[2] for r in results])
    percentages = np.array([r[3] for r in results])
    
    fig, ax = plt.subplots(figsize=(16, 10))
    x_pos = np.arange(len(class_names))
    bar_width = 0.78
    
    bars_small = ax.bar(x_pos, smalls, bar_width,
                        label='Small (<32²)', color='#A0C0C0', edgecolor='black', linewidth=0.6)
    ax.bar(x_pos, totals - smalls, bar_width, bottom=smalls,
           label='Medium + Large (≥32²)', color='#CBDEDE', edgecolor='black', linewidth=0.6)
    
    # Add percentage labels
    for bar, pct, small in zip(bars_small, percentages, smalls):
        height = bar.get_height()
        if height > 500:
            y_pos = bar.get_y() + height * 0.5
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='black')
    
    ax.set_yscale('log')
    
    # Use custom ylim if provided, otherwise auto-calculate
    if ylim_max:
        ax.set_ylim(50, ylim_max)
    else:
        ax.set_ylim(50, max(totals) * 1.2)
    
    class_labels = [f'{name}\n({int(small):,})' for name, small in zip(class_names, smalls)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels, rotation=90, ha='right', fontsize=9)
    ax.set_xlabel('Object Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of instances (log scale)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.95)
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10, subs='auto'))
    ax.grid(axis='y', which='major', alpha=0.4, linestyle='--')
    ax.grid(axis='y', which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_bubble_chart(results, title, output_path):
    if not results:
        print(f"Skipping bubble chart (no data): {output_path}")
        return
        
    class_names = [r[0] for r in results]
    totals = np.array([r[1] for r in results])
    percentages = np.array([r[3] for r in results])
    
    fig, ax = plt.subplots(figsize=(18, 10))
    x_pos = np.arange(len(class_names))
    
    # Normalize bubble sizes
    min_size, max_size = 50, 4000
    totals_norm = (totals - totals.min()) / (totals.max() - totals.min() + 1e-6)
    bubble_sizes = min_size + totals_norm ** 0.5 * (max_size - min_size)
    
    scatter = ax.scatter(x_pos, percentages, s=bubble_sizes, alpha=0.65,
                        c=percentages, cmap='YlOrRd', edgecolors='black', linewidth=1.2)
    
    # Add labels
    for i, (x, y, pct, size) in enumerate(zip(x_pos, percentages, percentages, bubble_sizes)):
        if size > 800:
            ax.text(x, y, f'{pct:.0f}%', ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color='black')
    
    ax.set_xlabel('Object Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Small Objects (<32²)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=9)
    ax.set_ylim(0, max(percentages) + 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Small Object %', fontsize=11, fontweight='bold')
    
    # Size legend - use smaller, fixed sizes for better readability
    legend_counts = [int(totals.min()), int(np.median(totals)), int(totals.max())]
    # Use small, medium, large fixed sizes instead of actual bubble sizes
    legend_sizes = [80, 200, 400]  # Much smaller and more distinguishable
    legend_elements = [plt.scatter([], [], s=s, c='gray', alpha=0.6, edgecolors='black',
                                  linewidth=1.2, label=f'{int(c):,} instances')
                      for s, c in zip(legend_sizes, legend_counts)]
    ax.legend(handles=legend_elements, loc='upper right', title='Total Count',
              fontsize=9, title_fontsize=10, framealpha=0.9, edgecolor='black',
              scatterpoints=1, markerscale=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def print_summary(results, title, top_n=30):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"{'Rank':>4} | {'Class':25} | {'Total':>10} | {'Small':>10} | {'Small %':>9}")
    print("-"*80)
    for i, (name, total, small, pct) in enumerate(results[:top_n], 1):
        print(f"{i:4d} | {name:25} | {total:10,d} | {small:10,d} | {pct:8.2f}%")
    print("="*80)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def compare_class_lists(coco_top30_names, oi_classes_csv):
    print("\n" + "="*80)
    print("COCO TOP-30 vs OPEN IMAGES V7 – CLASS OVERLAP CHECK")
    print("="*80)
    
    # Load Open Images classes (LabelName, DisplayName)
    oi_df = pd.read_csv(oi_classes_csv, header=None, names=["LabelName", "DisplayName"])
    oi_classes = set(oi_df["DisplayName"].str.strip().str.lower())
    coco_classes = {c.strip().lower() for c in coco_top30_names}
    
    matched = sorted(coco_classes & oi_classes)
    missing = sorted(coco_classes - oi_classes)
    
    print(f"COCO Top-30 classes        : {len(coco_classes)}")
    print(f"Open Images classes        : {len(oi_classes)}")
    print(f"Matched classes            : {len(matched)}")
    print(f"Missing in Open Images     : {len(missing)}\n")
    
    print("✓ Matched classes:")
    for c in matched:
        print(f"  - {c}")
    
    if missing:
        print("\n✗ Missing classes:")
        for c in missing:
            print(f"  - {c}")
    
    print("="*80)
    
    # Return original casing from COCO for use in filtering
    matched_original = {name for name in coco_top30_names if name.lower() in matched}
    return matched_original


def analyze_coco():
    print("\n" + "="*80)
    print("ANALYZING COCO DATASET")
    print("="*80)
    print("Loading COCO annotations...")
    train_data = load_json(COCO_ANN_TRAIN)
    val_data = load_json(COCO_ANN_VAL)
    cat_id_to_name = {c["id"]: c["name"] for c in train_data["categories"]}

    # Prepare three versions
    datasets = {
        "train": train_data["annotations"],
        "val": val_data["annotations"],
        "combined": train_data["annotations"] + val_data["annotations"],
    }

    results_dict = {}
    for split_name, annotations in datasets.items():
        print(f"\n→ Processing {split_name} ({len(annotations):,} annotations)")
        
        total, small = count_small_objects(annotations, SMALL_THRESH)
        
        results = create_results_table(total, small, cat_id_to_name)
        results.sort(key=lambda x: x[3], reverse=True)  # by % descending
        
        results_dict[split_name] = results
        
        top30 = results[:30]
        
        title_suffix = f"COCO {split_name.capitalize()} – Top 30 Classes by Small Object %"
        
        # Plot three chart types (fixed log lambda)
        for chart_type, plot_func in [
            ("proportion", plot_proportion_chart),
            ("log", lambda res, tit, pth: plot_log_chart(res, tit, pth, ylim_max=1_000_000)),
            ("bubble", plot_bubble_chart),
        ]:
            filename = get_plot_filename("coco", split_name, chart_type, RUNTIME_ID)
            path = os.path.join(COCO_OUT_DIR, filename)
            
            plot_func(
                top30,
                title_suffix,
                path
            )
        
        # Summary print
        print_summary(results, f"COCO {split_name.upper()} – TOP 30 SMALL OBJECT %")
        
        all_total = sum(total.values())
        all_small = sum(small.values())
        global_pct = all_small / all_total * 100 if all_total > 0 else 0
        print(f"Global ({split_name}): {all_total:,} total, {all_small:,} small ({global_pct:.2f}%)\n")

    # ────────────────────────────────────────────────
    # After all splits are done → create Train vs Val stacked plot
    # ────────────────────────────────────────────────
    new_title = "COCO Combined: Small Objects – Train vs Val Contribution"
    new_path = os.path.join(
        COCO_OUT_DIR,
        f"{RUNTIME_ID}_coco_combined_small_objects_trainval_stacked.png"
    )
    plot_train_val_small_stacked(results_dict, new_title, new_path)

    # Return the full dict for later use
    return results_dict


def analyze_open_images(coco_top30_names=None, oi_data=None):
    print("\n" + "="*80)
    print("ANALYZING OPEN IMAGES V7")
    if coco_top30_names:
        print("(Filtering for COCO Top 30 classes)")
    print("="*80)

    if oi_data is None:
        print("Loading class names...")
        classes_df = pd.read_csv(OI_CLASSES_CSV, header=None, names=["LabelName", "DisplayName"])
        mid_to_name = dict(zip(classes_df["LabelName"], classes_df["DisplayName"]))

        print("Loading detections...")
        dets_val   = pd.read_csv(OI_LABELS_VAL)
        dets_train = pd.read_csv(OI_LABELS_TRAIN)
        dets = pd.concat([dets_train, dets_val], ignore_index=True)
        dets = dets[dets["Confidence"] == 1].copy()
        dets["DisplayName"] = dets["LabelName"].map(mid_to_name)

        data_dirs = [
            os.path.join(OI_BASE_TRAIN, "data"),
            os.path.join(OI_BASE_VAL,   "data")
        ]
        img_sizes = load_oi_image_sizes(data_dirs)
    else:
        print("Reusing previously loaded data...")
        dets, img_sizes = oi_data

    # ────────────────────────────────────────────────
    # Split into train / val / combined
    # ────────────────────────────────────────────────
    dets_train = dets[dets["ImageID"].str.startswith("train")]
    dets_val   = dets[dets["ImageID"].str.startswith("val")]

    datasets = {
        "train":    dets_train,
        "val":      dets_val,
        "combined": dets,
    }

    results_dict = {}

    for split_name, df_split in datasets.items():
        print(f"\n→ Processing {split_name} ({len(df_split):,} boxes)")
        
        total, small = count_small_objects_oi(
            df_split,
            img_sizes,
            SMALL_THRESH,
            filter_classes=coco_top30_names
        )

        # Debug: show if Computer mouse exists before renaming ──
        if "Computer mouse" in total:
            print(f"DEBUG: Found 'Computer mouse' → {total['Computer mouse']:,} total instances")
        if "Mouse" in total:
            print(f"DEBUG: Found animal 'Mouse' → {total['Mouse']:,} total instances")

        # Build results list + special handling for mouse classes
        results = []
        for name in total:
            display_name = name
            skip = False

            if coco_top30_names is not None:   # only in filtered run
                if name == "mouse":             # after rename
                    display_name = "mouse"
                    print(f"DEBUG: Using renamed 'mouse' (Computer mouse) → {total[name]:,} instances")
                elif name == "Mouse":           # animal
                    skip = True
                    print(f"DEBUG: Skipping animal 'Mouse' ({total[name]:,} instances)")

            if skip:
                continue

            t = total[name]
            s = small[name]
            if t == 0:
                continue
            results.append((display_name, t, s, s / t * 100 if t > 0 else 0))
            
            
        results.sort(key=lambda x: x[3], reverse=True)
        results_dict[split_name] = results

        if not results:
            print(f"→ No objects found in {split_name}")
            continue

        if coco_top30_names:
            dir_path = os.path.join(OI_OUT_DIR, "comparison")
            prefix = f"{RUNTIME_ID}_coco_top30"
            titles = {
                "proportion": "Normalized small-object ratio for the 24 COCO-matched classes in OpenImages V7",
                "log":        "Log-scale absolute small-object counts for the 24 COCO-matched classes (in OpenImages V7)",
                "bubble":     f"Open Images {split_name.capitalize()} – Top 30 Small Object % – COCO Top 30 classes",
            }
        else:
            dir_path = OI_OUT_DIR
            prefix = RUNTIME_ID
            results = results[:30]  # only top 30 when not filtering
            base_title = f"Open Images {split_name.capitalize()} – Top 30 Small Object %"
            titles = {"proportion": base_title, "log": base_title, "bubble": base_title}

        for chart_type, plot_func in [
            ("proportion", plot_proportion_chart),
            ("log",        lambda r,t,o: plot_log_chart(r, t, o)),
            ("bubble",     plot_bubble_chart),
        ]:
            fname = get_plot_filename("openimages", split_name, chart_type, prefix)
            path = os.path.join(dir_path, fname)
            plot_func(results, titles[chart_type], path)

        print_summary(results, f"OPEN IMAGES {split_name.upper()} – TOP 30 SMALL %")
        
        all_t = sum(total.values())
        all_s = sum(small.values())
        print(f"Global ({split_name}): {all_t:,} total, {all_s:,} small ({all_s/all_t*100:.2f}%)\n")

    if oi_data is None:
        return results_dict, (dets, img_sizes)
    else:
        return results_dict


def compare_datasets(coco_top30, oi_results):
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)
    
    if not oi_results:
        print("Warning: No Open Images data to compare - skipping comparison")
        return
    
    # Match classes
    coco_dict = {r[0]: r for r in coco_top30}
    oi_dict = {r[0]: r for r in oi_results}
    
    comparison = []
    for name in coco_dict:
        if name in oi_dict:
            c = coco_dict[name]
            o = oi_dict[name]
            comparison.append((name, c[1], c[2], c[3], o[1], o[2], o[3]))
    
    comparison.sort(key=lambda x: x[3], reverse=True)
    
    # Side-by-side comparison for ALL matched classes (up to 24)
    class_names = [r[0] for r in comparison]
    coco_pct = np.array([r[3] for r in comparison])
    oi_pct = np.array([r[6] for r in comparison])

    fig, ax = plt.subplots(figsize=(20, 12))  # wider + taller for 24 classes
    x_pos = np.arange(len(class_names))
    bar_width = 0.35

    bars_coco = ax.bar(x_pos - bar_width/2, coco_pct, bar_width,
                       label='COCO % small', color='#A0C0C0', edgecolor='black')
    bars_oi = ax.bar(x_pos + bar_width/2, oi_pct, bar_width,
                     label='Open Images % small', color='#CBDEDE', edgecolor='black')

    # Add percentage labels on top of bars
    for bar in bars_coco:
        height = bar.get_height()
        if height > 2:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_oi:
        height = bar.get_height()
        if height > 2:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=10)
    ax.set_xlabel('Object Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Small Object % (<32×32 px)', fontsize=14, fontweight='bold')
    ax.set_title('COCO vs Open Images V7: Small Object % Comparison\n'
                 f'(Top 24 Matched Classes from COCO)',
                 fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    comparison_dir = os.path.join(OI_OUT_DIR, "comparison")
    comparison_path = os.path.join(
        comparison_dir,
        f"{RUNTIME_ID}_coco_vs_oi_top24_comparison.png"  # new name
    )
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison chart (24 classes): {comparison_path}")
        
    # Print comparison table
    print("\n" + "="*120)
    print(f"COCO VS OPEN IMAGES - COMPARISON TABLE ({len(comparison)} Matched Classes)")
    print("="*120)
    print(f"{'Rank':>4} | {'Class':25} | {'COCO Total':>10} | {'COCO Small':>10} | {'COCO %':>8} | "
          f"{'OI Total':>10} | {'OI Small':>10} | {'OI %':>8}")
    print("-"*120)
    for i, (name, c_tot, c_sm, c_pct, o_tot, o_sm, o_pct) in enumerate(comparison, 1):
        print(f"{i:4d} | {name:25} | {c_tot:10,d} | {c_sm:10,d} | {c_pct:8.2f}% | "
              f"{o_tot:10,d} | {o_sm:10,d} | {o_pct:8.2f}%")
    print("="*120)


# ============================================================================
# RUN ALL ANALYSES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATASET ANALYSIS: COCO & OPEN IMAGES V7")
    print("Small Object Threshold: 32×32 pixels")
    print("="*80)
    
    # Analyze COCO → returns dict with train/val/combined
    # ────────────────────────────────────────────────
    results_dict_coco = analyze_coco()

    # We still need the set of top-30 class names from combined for matching
    coco_top30_combined = results_dict_coco["combined"][:30]
    coco_top30_names = {r[0] for r in coco_top30_combined}

    # Check which COCO classes exist in Open Images
    matched_classes = compare_class_lists(coco_top30_names, OI_CLASSES_CSV)

    # ────────────────────────────────────────────────
    # Analyze Open Images (full run – gets all classes)
    # ────────────────────────────────────────────────
    results_dict_oi_full, oi_data = analyze_open_images()

    # ────────────────────────────────────────────────
    # Analyze Open Images filtered to matched COCO classes
    # ────────────────────────────────────────────────
    results_dict_oi_matched = None
    if matched_classes:
        print(f"\nAnalyzing {len(matched_classes)} matched classes in Open Images...")
        results_dict_oi_matched = analyze_open_images(matched_classes, oi_data)

    # ────────────────────────────────────────────────
    # Final comparison – using combined splits
    # ────────────────────────────────────────────────
    if matched_classes and results_dict_oi_matched:
        print(f"\nCreating comparison using combined train+val splits...")

        coco_combined_top30 = results_dict_coco["combined"][:30]
        oi_matched_combined = results_dict_oi_matched["combined"]

        # Optional: filter to classes actually present in both combined results
        coco_names_present = {r[0] for r in coco_combined_top30}
        oi_names_present   = {r[0] for r in oi_matched_combined}
        common_names = coco_names_present & oi_names_present
        
        # Force comparison with original lists (for debugging)
        compare_datasets(coco_combined_top30, oi_matched_combined)
        
    else:
        print("\nNo matching classes found - skipping comparison")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)