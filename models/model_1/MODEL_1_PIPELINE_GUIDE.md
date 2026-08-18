# Model 1 (Fish Freshness Classifier) — Pipeline Guide & Analysis
### Proyek NusaQC · COMPFEST 18 AIC (Smart Manufacturing Track)

Dokumen ini menjelaskan alur kerja (*pipeline*), panduan eksekusi di Kaggle, interpretasi artefak visual, serta analisis ilmiah dari hasil pengujian model **NusaQC Model 1: Freshness Engine (MobileNetV3 INT8 ONNX)**.

---

## 1. Arsitektur Single-File All-in-One Pipeline

Seluruh alur dari **EDA**, **Dual-Split**, **Class-Weighted Training**, **Secondary FFE Validation**, **Visualisasi**, **ONNX Export (Opset 17)**, hingga **Benchmarking Latensi CPU** dikemas dalam satu file:
👉 **[`01_model1_full_pipeline.py`](file:///d:/main/Documents/explore/compe/hackhathon/AIC/models/model_1/01_model1_full_pipeline.py)**

### Keunggulan Utama Pipeline:
1. **Ultra-Fast In-Memory Preload (`FastRAMFishDataset`):** Mengeliminasi bottleneck disk I/O Kaggle dengan me-resize dan menyimpan seluruh dataset ke RAM saat inisialisasi (~15 detik). Seluruh training 12 epoch selesai dalam < 2 menit di GPU T4/P100.
2. **Autodiscovery Dataset:** Mendukung path input Kaggle resmi, symlink, maupun direktori lokal secara otomatis.
3. **Class-Weighted CrossEntropyLoss:** Menyeimbangkan bobot kelas minoritas (Grade A hanya 15.8% vs Grade C 47.2%).
4. **Anti-Leakage Spatial Augmentation:** Regularisasi `RandomErasing` (Cutout) dan `ColorJitter` untuk mencegah CNN overfit pada latar meja laboratorium DaFiF.
5. **Fixed ONNX Opset 17 & Dynamic INT8 Quantization:** Kompatibel dengan PyTorch 2.x, terbebas dari error Dynamo/onnxscript, dan menghasilkan INT8 yang tervalidasi.
6. **Edge Latency Benchmarking:** Pengujian latensi CPU otomatis (ms/frame dan FPS) untuk validasi kesiapan edge computing (Raspberry Pi 5).

---

## 2. Peta Artefak Visual yang Dihasilkan

Setelah script dijalankan, folder `/kaggle/working/output_model1/` akan berisi artefak beresolusi tinggi (300 DPI) berikut yang siap dilampirkan pada Proposal:

| Nama File Artefak | Deskripsi & Isi Visual | Kegunaan dalam Proposal |
| :--- | :--- | :--- |
| `eda_dataset_distribution.png` | 4 chart: Distribusi Kelas DaFiF, Distribusi Kelas per Spesies, Degradasi Mutu per Hari (Day 1–11), dan Perbandingan DaFiF vs FFE. | **Bab 2: Data Acquisition & EDA** |
| `eda_sample_images_by_grade.png` | Grid foto asli ikan DaFiF untuk Grade A, Grade B, dan Grade C beserta metadata spesies, hari, dan sesi. | **Bab 2 / Lampiran Visualisasi Citra** |
| `training_curves_comparison.png` | Kurva perbandingan Loss, Accuracy (%), dan Macro F1 per epoch untuk Random Split vs Grouped Split. | **Bab 3: Eksperimen & Analisis Konvergensi** |
| `confusion_matrices_all.png` | Matriks kebingungan (Raw Counts & Normalized %) untuk Random Split, Grouped Split, dan FFE Cross-Validation. | **Bab 3: Evaluasi Kinerja Model & Safety Critical Metric** |
| `sample_test_predictions_grid.png` | Grid prediksi citra test set yang menampilkan probabilitas kepercayaan dan status *CORRECT* (Hijau) / *MISCLASSIFIED* (Merah). | **Bab 3: Error Analysis & Validasi Kualitatif** |
| `mobilenetv3_freshness.onnx` | Model ONNX Float32 (Opset 17) tervalidasi (~3.5 MB). | Model Artefak untuk FastAPI Backend |
| `mobilenetv3_freshness_int8.onnx` | Model ONNX INT8 Terkuantisasi (~1.0 MB). | Model Artefak untuk Edge Device (Raspberry Pi 5) |

---

## 3. Bedah Ilmiah Hasil Evaluasi (Data Leakage & Domain Shift)

### A. Random Split (99.48% Acc, 0.9939 F1) — *Spatiotemporal Leakage*
* **Penyebab:** Pada dataset DaFiF, 10 ekor ikan dipotret berulang kali dalam satu sesi. Pada Random Split, foto ikan yang sama pada pencahayaan dan sudut serupa tersebar di train dan test set.
* **Kesimpulan:** Akurasi 99.48% adalah hasil *artificially inflated* karena model menghafal spesimen dan latar laboratorium.

### B. Grouped Split by Day/Session (80.00% Acc, 0.7496 F1) — *Generalisasi Realistis*
* **Penyebab:** Ketika sesi dipisahkan total, model diuji pada sesi/hari baru yang belum pernah dilihat saat training.
* **Kesimpulan:** 80.00% adalah performa jujur dan realistis model DaFiF. Inilah nilai yang harus diangkat dalam proposal sebagai benchmark integritas engineering.

### C. Secondary Validation FFE (36.29% Acc, 0.2966 F1) — *Modality & Domain Gap*
* **Penyebab:** DaFiF adalah foto *whole-body*, sedangkan FFE adalah *macro close-up* murni pada kornea mata ikan. Fitur tubuh/insang/latar yang dipelajari dari DaFiF tidak ada pada FFE.
* **Kesimpulan:** Menjadi bukti ilmiah kuat di proposal bahwa model conveyor whole-body memerlukan kamera bersudut makro khusus jika ingin melakukan inspeksi berbasis mata murni.

---

## 4. Panduan Eksekusi di Kaggle

1. Buka [Kaggle Notebook](https://www.kaggle.com/).
2. Pasang Accelerator: **GPU T4 / P100**.
3. Tambahkan 2 dataset input:
   - `raykapranandita/dataset-for-fishs-freshness-problems`
   - `raykapranandita/the-freshness-of-the-fish-eyes-dataset-ffe`
4. Copy-paste isi [`01_model1_full_pipeline.py`](file:///d:/main/Documents/explore/compe/hackhathon/AIC/models/model_1/01_model1_full_pipeline.py) ke dalam notebook cell.
5. Jalankan (**Run All**). Waktu eksekusi rata-rata: **~2 menit**.
6. Unduh folder `/kaggle/working/output_model1/` untuk mengambil seluruh grafik visual dan bobot model `.onnx`.
