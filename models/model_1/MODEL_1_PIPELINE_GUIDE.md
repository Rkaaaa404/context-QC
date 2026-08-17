# Model 1 (Fish Freshness Classifier) — Execution Guide & Kaggle Pipeline

Dokumen ini berisi panduan alur kerja (*step-by-step pipeline*) untuk melatih **Model 1: Fish Freshness Classifier (MobileNetV3-Small INT8 ONNX)** pada platform Kaggle tanpa menguras kuota GPU.

---

## 1. Strategi Efisiensi Kuota GPU Kaggle (2-Step Notebook Architecture)

Untuk menghindari pemborosan kuota GPU Kaggle akibat proses unzipping, pemindaian direktori yang lambat, atau *resizing* berulang setiap epoch:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PREPROCESSING NOTEBOOK (Kaggle CPU / Local)                    │
│ Input  : Raw DaFiF & FFE Datasets                                      │
│ Action : Map Grade A/B/C, Resize 224x224, Train/Val/Test Split          │
│ Output : `/kaggle/working/nusaqc_freshness_processed.zip`              │
│          -> Simpan sebagai New Kaggle Dataset (`nusaqc-freshness-proc`) │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (Attach Dataset ke Notebook 2)
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: TRAINING NOTEBOOK (Kaggle GPU T4 / P100)                       │
│ Input  : `/kaggle/input/nusaqc-freshness-proc`                         │
│ Action : PyTorch Training (MobileNetV3), Fine-Tuning, Metrics Eval    │
│ Output : `mobilenetv3_freshness.onnx` & `mobilenetv3_freshness_int8.onnx`│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Definisi Path Dataset di Kaggle

### Input Dataset Path (Raw):
* **DaFiF Base Path:**
  `/kaggle/input/datasets/raykapranandita/dataset-for-fishs-freshness-problems/Dataset for Fishs Freshness Problems`
* **FFE Base Path:**
  `/kaggle/input/datasets/raykapranandita/the-freshness-of-the-fish-eyes-dataset-ffe`

### Rules Pemetaan Grade (SNI 2729:2013 & Organoleptik):
1. **DaFiF Dataset:**
   * `Day 1` & `Day 2` $\rightarrow$ **Grade A** (Freshness score 8.0 – 9.0)
   * `Day 3` s/d `Day 6` $\rightarrow$ **Grade B** (Freshness score 6.0 – 7.9)
   * `Day 7` s/d `Day 11` $\rightarrow$ **Grade C** (Freshness score 1.0 – 5.9)
2. **FFE Dataset:**
   * Folder berakhiran `- Highly Fresh` $\rightarrow$ **Grade A**
   * Folder berakhiran `- Fresh` $\rightarrow$ **Grade B**
   * Folder berakhiran `- Not Fresh` $\rightarrow$ **Grade C**

---

## 3. Struktur File Script Python

Semua script dibuat modular di direktori `model-1/`:
1. [`01_preprocess_dataset.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1/01_preprocess_dataset.py) — Script Preprocessing & Data Harmonization (Kaggle CPU)
2. [`02_train_mobilenetv3.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1/02_train_mobilenetv3.py) — Script Training, Evaluation & ONNX Quantization (Kaggle GPU)

---

## 4. Langkah Imputasi & Eksekusi di Kaggle

### Langkah Step 1:
1. Buat **Kaggle Notebook Baru (CPU Accelerator)**.
2. Hubungkan 2 dataset input Kaggle kamu:
   - `dataset-for-fishs-freshness-problems`
   - `the-freshness-of-the-fish-eyes-dataset-ffe`
3. Paste / Jalankan kode dari [`01_preprocess_dataset.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1/01_preprocess_dataset.py).
4. Klik **Save Version** $\rightarrow$ **Save & Run All (Commit)**.
5. Setelah selesai, buka Output Notebook, klik **"Save as Dataset"** dengan nama dataset: `nusaqc-freshness-processed`.

### Langkah Step 2:
1. Buat **Kaggle Notebook Baru (GPU T4 / P100 Accelerator)**.
2. Attach dataset buatanmu: `nusaqc-freshness-processed`.
3. Paste / Jalankan kode dari [`02_train_mobilenetv3.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1/02_train_mobilenetv3.py).
4. Training akan berjalan cepat (100% efisiensi GPU).
5. Unduh file `mobilenetv3_freshness_int8.onnx` untuk diintegrasikan ke FastAPI & Raspberry Pi 5!
