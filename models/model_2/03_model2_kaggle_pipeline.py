# %% [markdown]
# # NusaQC Model 2: Fish Defect Detector (YOLOv8s)
# Harmonisasi dataset Roboflow Fish Disease, training YOLOv8s, evaluasi, dan ekspor ONNX.

# %% [markdown]
# # 1. Setup Environment

# %%
import os, sys, shutil, yaml
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import ultralytics
except ImportError:
    os.system('pip install -q ultralytics')

import torch
torch_version = torch.__version__
has_cuda = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if has_cuda else 'CPU'
print(f"PyTorch {torch_version} | CUDA: {has_cuda} ({gpu_name})")

# --- Paths ---
DATASET_ROOT = Path("/kaggle/input/datasets/raykapranandita/roboflow-fish-disease")
if not DATASET_ROOT.exists():
    DATASET_ROOT = Path("models/datasets/model-2/roboflow-fish-disease")

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./runs_model2_workspace")
CLEAN_DATASET_DIR = WORKING_DIR / "nusaqc_clean_dataset"
WORKING_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DATASET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset Root : {DATASET_ROOT.resolve()}")
print(f"Working Dir  : {WORKING_DIR.resolve()}")
print(f"Clean Dataset: {CLEAN_DATASET_DIR.resolve()}")

for split in ['train', 'valid', 'test']:
    imgs = len(list((DATASET_ROOT / split / 'images').glob('*.*'))) if (DATASET_ROOT / split / 'images').exists() else 0
    lbls = len(list((DATASET_ROOT / split / 'labels').glob('*.txt'))) if (DATASET_ROOT / split / 'labels').exists() else 0
    print(f"  [{split:<5}] Images: {imgs:<5} | Labels: {lbls:<5}")

# %% [markdown]
# # 2. EDA: Analisis Dataset Mentah

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

RAW_CLASS_NAMES = ['BDA', 'BGD', 'BRD', 'FDS', 'HF', 'PD', 'WTD']
NUSAQC_CLASSES = {0: "sisik_sisa", 1: "warna_abnormal", 2: "luka_robekan", 3: "lendir_berlebih"}
NUSAQC_CLASS_NAMES = list(NUSAQC_CLASSES.values())

def collect_raw_stats(dataset_root):
    records, anomalies = [], []
    for split in ['train', 'valid', 'test']:
        lbl_dir = dataset_root / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    area = w * h
                    is_valid = all([0 <= xc <= 1, 0 <= yc <= 1, 0 < w <= 1, 0 < h <= 1])
                    if not is_valid or area <= 1e-6:
                        anomalies.append({'split': split, 'file': lbl_file.name, 'line': idx+1,
                                          'reason': 'out_of_bounds' if not is_valid else 'zero_area'})
                    records.append({
                        'split': split, 'class_id': cls_id,
                        'class_name': RAW_CLASS_NAMES[cls_id] if cls_id < len(RAW_CLASS_NAMES) else f"C{cls_id}",
                        'xc': xc, 'yc': yc, 'width': w, 'height': h,
                        'area': area, 'aspect_ratio': w / (h + 1e-6)
                    })
                except ValueError:
                    continue
    df = pd.DataFrame(records)
    print(f"Total anotasi: {len(df)} | Anomali: {len(anomalies)}")
    return df, anomalies

def visualize_eda(df):
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.countplot(data=df, x='class_name', hue='split', ax=axes[0,0], palette='viridis', order=RAW_CLASS_NAMES)
    axes[0,0].set_title("Frekuensi Kelas per Split"); axes[0,0].tick_params(axis='x', rotation=30)
    sns.boxplot(data=df, x='class_name', y='area', hue='class_name', ax=axes[0,1], palette='crest', order=RAW_CLASS_NAMES, legend=False)
    axes[0,1].set_title("Area BBox per Kelas"); axes[0,1].tick_params(axis='x', rotation=30)
    sns.boxplot(data=df, x='class_name', y='aspect_ratio', hue='class_name', ax=axes[1,0], palette='magma', order=RAW_CLASS_NAMES, legend=False)
    axes[1,0].set_title("Aspect Ratio per Kelas"); axes[1,0].set_yscale('log'); axes[1,0].tick_params(axis='x', rotation=30)
    counts = df['class_name'].value_counts().reindex(RAW_CLASS_NAMES).fillna(0)
    axes[1,1].bar(counts.index, counts.values, color='teal')
    axes[1,1].set_title("Total BBox per Kelas"); axes[1,1].tick_params(axis='x', rotation=30)
    for i, v in enumerate(counts.values):
        axes[1,1].text(i, v + max(counts.values)*0.01, str(int(v)), ha='center', fontsize=9)
    plt.tight_layout(); plt.show()

def visualize_samples(dataset_root, n=4):
    img_dir = dataset_root / "train" / "images"
    lbl_dir = dataset_root / "train" / "labels"
    imgs = list(img_dir.glob("*.*"))[:n]
    if not imgs:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(4*len(imgs), 4))
    if len(imgs) == 1:
        axes = [axes]
    for ax, ip in zip(axes, imgs):
        img = cv2.imread(str(ip))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ih, iw = img.shape[:2]
        lp = lbl_dir / f"{ip.stem}.txt"
        if lp.exists():
            for line in open(lp):
                p = line.strip().split()
                if len(p) >= 5:
                    cid = int(p[0]); xc,yc,bw,bh = map(float, p[1:5])
                    x1,y1 = int((xc-bw/2)*iw), int((yc-bh/2)*ih)
                    x2,y2 = int((xc+bw/2)*iw), int((yc+bh/2)*ih)
                    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.putText(img, RAW_CLASS_NAMES[cid] if cid<len(RAW_CLASS_NAMES) else str(cid),
                                (x1,max(y1-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        ax.imshow(img); ax.set_title(ip.name[:18]+"...", fontsize=9); ax.axis('off')
    plt.tight_layout(); plt.show()

df_raw, anomalies = collect_raw_stats(DATASET_ROOT)
visualize_eda(df_raw)
visualize_samples(DATASET_ROOT)

# %% [markdown]
# # 3. Harmonisasi Label: 7 Kelas Roboflow -> 4 Kelas NusaQC
# BDA/BGD/BRD -> warna_abnormal, FDS -> luka_robekan, HF -> skip, PD -> sisik_sisa, WTD -> lendir_berlebih

# %%
RAW_TO_NUSAQC = {
    0: 1,   # BDA -> warna_abnormal
    1: 1,   # BGD -> warna_abnormal
    2: 1,   # BRD -> warna_abnormal
    3: 2,   # FDS -> luka_robekan
    4: -1,  # HF  -> skip (healthy)
    5: 0,   # PD  -> sisik_sisa
    6: 3    # WTD -> lendir_berlebih
}

def harmonize_dataset(src_root, dst_root):
    remapped, skipped = 0, 0
    for split in ['train', 'valid', 'test']:
        src_img = src_root / split / "images"
        src_lbl = src_root / split / "labels"
        dst_img = dst_root / split / "images"
        dst_lbl = dst_root / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        if not src_img.exists():
            continue
        for img_path in src_img.glob("*.*"):
            shutil.copy(img_path, dst_img / img_path.name)
            lbl_path = src_lbl / f"{img_path.stem}.txt"
            lines = []
            if lbl_path.exists():
                for line in open(lbl_path, 'r', encoding='utf-8'):
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    try:
                        raw_cls = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        xc, yc = max(0, min(1, xc)), max(0, min(1, yc))
                        w, h = max(0.001, min(1, w)), max(0.001, min(1, h))
                        if w*h <= 1e-6:
                            skipped += 1; continue
                        nq = RAW_TO_NUSAQC.get(raw_cls, -1)
                        if nq == -1:
                            skipped += 1; continue
                        lines.append(f"{nq} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                        remapped += 1
                    except ValueError:
                        continue
            with open(dst_lbl / f"{img_path.stem}.txt", 'w', encoding='utf-8') as f:
                f.writelines(lines)
    print(f"Harmonisasi: {remapped} bbox remapped, {skipped} skipped")

def create_data_yaml(dst_root, output_dir):
    yaml_path = output_dir / "data.yaml"
    cfg = {
        'path': dst_root.resolve().as_posix(),
        'train': (dst_root / "train" / "images").resolve().as_posix(),
        'val': (dst_root / "valid" / "images").resolve().as_posix(),
        'test': (dst_root / "test" / "images").resolve().as_posix(),
        'nc': len(NUSAQC_CLASS_NAMES),
        'names': NUSAQC_CLASS_NAMES
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"data.yaml -> {yaml_path.resolve()}")
    return yaml_path

harmonize_dataset(DATASET_ROOT, CLEAN_DATASET_DIR)
DATA_YAML = create_data_yaml(CLEAN_DATASET_DIR, WORKING_DIR)

# %% [markdown]
# # 4. Training YOLOv8s

# %%
from ultralytics import YOLO

HPARAMS = {
    'epochs': 100, 'batch': 16, 'imgsz': 640,
    'optimizer': 'AdamW', 'lr0': 0.001, 'lrf': 0.01,
    'momentum': 0.937, 'weight_decay': 0.001, 'warmup_epochs': 5.0,
    'mosaic': 1.0, 'close_mosaic': 15, 'mixup': 0.15, 'copy_paste': 0.2,
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
    'scale': 0.5, 'fliplr': 0.5, 'flipud': 0.1,
    'translate': 0.2, 'degrees': 10.0,
    'box': 7.5, 'cls': 1.5, 'dfl': 1.5, 'patience': 30
}

print("Hyperparameters:")
for k, v in HPARAMS.items():
    print(f"  {k:<15}: {v}")

model = YOLO("yolov8s.pt")
RUN_DIR = WORKING_DIR / "runs_model2" / "nusaqc_yolov8n_defect"

print("\nStarting training...")
train_results = model.train(
    data=str(DATA_YAML),
    project=str(WORKING_DIR / "runs_model2"),
    name="nusaqc_yolov8n_defect",
    save=True, plots=True, exist_ok=True,
    **HPARAMS
)

# %% [markdown]
# # 5. Evaluasi & Ekspor ONNX

# %%
def plot_results(run_dir):
    csv = run_dir / "results.csv"
    if not csv.exists():
        print(f"results.csv tidak ditemukan di {run_dir}"); return
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if 'train/box_loss' in df.columns:
        for col, lbl in [('train/box_loss','Box'),('train/cls_loss','Cls'),('train/dfl_loss','DFL')]:
            axes[0,0].plot(df['epoch'], df[col], label=lbl)
        axes[0,0].set_title("Train Loss"); axes[0,0].legend(); axes[0,0].grid(True, ls='--', alpha=0.6)
    if 'val/box_loss' in df.columns:
        for col, lbl, c in [('val/box_loss','Box','red'),('val/cls_loss','Cls','orange'),('val/dfl_loss','DFL','purple')]:
            axes[0,1].plot(df['epoch'], df[col], label=lbl, color=c)
        axes[0,1].set_title("Val Loss"); axes[0,1].legend(); axes[0,1].grid(True, ls='--', alpha=0.6)
    if 'metrics/mAP50(B)' in df.columns:
        axes[1,0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='green', lw=2)
        axes[1,0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='blue', lw=2)
        axes[1,0].set_title("mAP"); axes[1,0].legend(); axes[1,0].grid(True, ls='--', alpha=0.6)
    if 'metrics/precision(B)' in df.columns:
        axes[1,1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='teal')
        axes[1,1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='magenta')
        axes[1,1].set_title("Precision/Recall"); axes[1,1].legend(); axes[1,1].grid(True, ls='--', alpha=0.6)
    plt.tight_layout(); plt.show()

plot_results(RUN_DIR)

# Load best model for eval
best_pt = RUN_DIR / "weights" / "best.pt"
eval_model = model if model else None
if best_pt.exists():
    eval_model = YOLO(str(best_pt))
    print(f"Loaded best checkpoint: {best_pt}")

if eval_model:
    print("\nEvaluating on test split...")
    metrics = eval_model.val(data=str(DATA_YAML), split='test')
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")

    if hasattr(metrics.box, 'maps') and len(metrics.box.maps) == len(NUSAQC_CLASS_NAMES):
        plt.figure(figsize=(10, 5))
        bars = plt.bar(NUSAQC_CLASS_NAMES, metrics.box.maps, color='darkcyan')
        plt.title("mAP@50 per Kelas NusaQC (Test)"); plt.ylim(0, 1.05)
        for b in bars:
            plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.3f}", ha='center', fontsize=9)
        plt.tight_layout(); plt.show()

    # Export ONNX
    onnx_path = eval_model.export(format="onnx", imgsz=640, simplify=True)
    print(f"ONNX exported: {onnx_path}")

    # Copy outputs to easy-access location
    out_dir = WORKING_DIR / "MODEL_OUTPUTS"
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy(best_pt, WORKING_DIR / "best.pt")
        shutil.copy(best_pt, out_dir / "best_seed_model.pt")
        print(f"  ✓ best.pt -> {WORKING_DIR / 'best.pt'}")
    if onnx_path and Path(onnx_path).exists():
        shutil.copy(onnx_path, WORKING_DIR / "model2_defect_detector.onnx")
        shutil.copy(onnx_path, out_dir / "model2_defect_detector.onnx")
        print(f"  ✓ ONNX -> {WORKING_DIR / 'model2_defect_detector.onnx'}")
    print(f"\n✨ All outputs in: {out_dir.resolve()}")
