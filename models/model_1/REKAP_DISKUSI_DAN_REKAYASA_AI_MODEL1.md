# Rekapitulasi Diskusi & Keputusan Rekayasa AI (Model 1 — NusaQC)
### COMPFEST 18 · 2026 · Smart Manufacturing

---

## 1. Context & Scope Framework NusaQC
Dokumen ini merangkum seluruh diskusi, evaluasi jurnal, analisis pemenang terdahulu, dan keputusan arsitektur rekayasa AI untuk **Model 1 (Fish Freshness Engine)** pada sistem **NusaQC**.

* **Peran Sistem:** Machine Vision & Automated Inspector untuk Unit Pengolahan Ikan (UPI) ekspor.
* **Target Environment:** Conveyor Belt sortasi bergerak dengan edge computing device (Raspberry Pi 5 CPU).
* **Core Rule Compliance:** Menjaga kesesuaian dengan rulebook Penyisihan COMPFEST 18 (Synchronous Snapshot, `docker-compose up`, CPU-only ONNX inference).

---

## 2. Benchmark Jurnal Ilmiah Terkini (Hoang et al., 2026)
* **Referensi:** *Deep feature optimization for enhanced fish freshness assessment* (Elsevier - Ecological Informatics 95, March 2026).
* **Temuan Jurnal:**
  * Jurnal mengevaluasi dataset **FFE (Freshness of Fish Eyes)** menggunakan pendekatan *Hybrid Deep Feature Extraction + Embedded Feature Selection (LGBM) + Random Forest Classifier*.
  * Hasil terbaik mencapai akurasi **85.99%** (meningkat 8–22% dibanding baseline CNN konvensional).
* **Penerapan pada NusaQC:**
  * Jurnal ini dijadikan referensi ilmiah utama pada **Sub-bab 2.3 Proposal & Engineering Decision Records (EDR)**.
  * Membuktikan bahwa pemangkasan dimensi fitur (*feature optimization*) dan kuantisasi INT8 ONNX sangat efektif untuk klasifikasi kesegaran fisik.

---

## 3. Evaluasi Dataset DaFiF vs FFE

| Parameter | Dataset DaFiF | Dataset FFE |
| :--- | :--- | :--- |
| **Karakteristik Visual** | Foto kepala & tubuh ikan utuh (*full frame*) | Foto *crop close-up* khusus pupil/kornea mata |
| **Ukuran Data** | 2.536 foto (Mackerel: 859, Tilapia: 840, Tuna: 837) + 9.401 data sensor gas E-Nose | 2.199 foto (3 spesies Tier 1 terpilih) / 4.390 (8 spesies) |
| **Standar Acuan** | SNI 2729:2013 Organoleptik (Mata, Insang, Lendir, Bau, Daging, Tekstur) | Tingkat kejernihan mata (*Eye-clarity*) |
| **Relevansi Conveyor** | **Sangat Tinggi** (kamera conveyor memotret dari atas/samping) | Sedang (membutuhkan makro lensa khusus) |

### Keputusan Strategis Dataset:
* **DaFiF dijadikan dataset utama Model 1 MVP** karena merepresentasikan foto ikan di conveyor belt secara nyata sesuai standar SNI.
* **FFE** disimpan sebagai data pembanding/eksperimen sekunder.

---

## 4. Arsitektur Pipeline & Efisiensi Kaggle GPU (2-Step Workflow)

Untuk mencegah pemborosan kuota GPU Kaggle (30 jam/minggu) akibat *unzipping* dan *resizing* berulang tiap epoch, pipeline dipisah menjadi 2 script:

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PREPROCESSING & HARMONIZATION (`01_preprocess_dataset.py`)     │
│ Runtime: Kaggle CPU / Local                                            │
│ Output : `/kaggle/working/nusaqc_freshness_processed.zip`              │
│ Action : Di-save sebagai New Kaggle Dataset (`nusaqc-freshness-proc`) │
└────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (Attach Dataset ke Notebook 2)
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: TRAINING & ONNX EXPORT (`02_train_mobilenetv3.py`)             │
│ Runtime: Kaggle GPU T4 / P100                                          │
│ Output : `mobilenetv3_freshness.onnx` & `mobilenetv3_freshness_int8.onnx`│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Status Arsitektur Model: Baseline vs Opsi Wacana Iterasi

### Status Saat Ini (Baseline Aktif):
* **Direct 3-Class Classification (`Grade_A`, `Grade_B`, `Grade_C`):**
  * Model MobileNetV3-Small dilatih langsung memprediksi 3 kelas mutu SNI 2729:2013.
  * Hasil pengujian test set saat ini: **Accuracy 99.48% & Macro F1 0.9939**.

### Wacana Diskusi Lanjutan (Proposed Concept - Belum Diimplementasikan):
Jika di kemudian hari ingin mengembangkan variasi model agar terkesan lebih ilmiah dan memberikan estimasi *shelf-life* di UI:

```
                  KONSEP WACANA ITERASI (PROPOSED CONCEPT)
                  ─────────────────────────────────────────

  [ Kamera Conveyor ]
           │ (Single Frame Snapshot)
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  AI MODEL 1: Freshness Day Estimator (Wacana)           │
  │  Output   : Estimasi Hari Storage Es (d_hat ∈ [1..11])  │
  └─────────────────────────────────────────────────────────┘
           │
           ▼ (Continuous Value: e.g., d_hat = 4.2 hari)
  ┌─────────────────────────────────────────────────────────┐
  │  BACKEND FASTAPI: SNI 2729:2013 Business Logic Engine   │
  │  Logic: Pemetaan nilai hari d_hat ke Grade A / B / C    │
  └─────────────────────────────────────────────────────────┘
           │
           ▼
  [ DASHBOARD UI & HARDWARE RELAY ]
```

*Catatan: Konsep di atas masih dalam tahap opsi diskusi dan belum diubah di script kode utama.*

---

## 6. Berkas Kode yang Telah Diperbarui

Seluruh berkas tersedia di direktori [`main/model-1/`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1):

1. **[`01_preprocess_dataset.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1/01_preprocess_dataset.py):** Clean Python script dengan penanda cell `# %%` untuk Kaggle CPU preprocessing.
2. **[`02_train_mobilenetv3.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1/02_train_mobilenetv3.py):** Clean Python script dengan penanda cell `# %%` dan pencarian path otomatis yang mendukung `/kaggle/input/datasets/raykapranandita/nusa-qc`.
3. **[`MODEL_1_PIPELINE_GUIDE.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/main/model-1/MODEL_1_PIPELINE_GUIDE.md):** Dokumentasi ringkas penggunaan script di Kaggle.

---

*Dokumen ini dibuat otomatis sebagai rekapitulasi teknis resmi tim AI NusaQC.*
