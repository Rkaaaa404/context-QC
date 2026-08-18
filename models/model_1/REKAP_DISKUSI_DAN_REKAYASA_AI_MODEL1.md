# Rekapitulasi Diskusi & Keputusan Rekayasa AI (Model 1 — NusaQC)
### COMPFEST 18 · 2026 · Smart Manufacturing Track

Dokumen ini merangkum seluruh diskusi, evaluasi jurnal, analisis pemenang terdahulu, bedah data leakage, dan keputusan arsitektur rekayasa AI untuk **Model 1 (Fish Freshness Engine)** pada sistem **NusaQC**.

---

## 1. Context & Scope Framework NusaQC
* **Peran Sistem:** Machine Vision & Automated Inspector untuk Unit Pengolahan Ikan (UPI) ekspor.
* **Target Environment:** Conveyor Belt sortasi bergerak dengan edge computing device (Raspberry Pi 5 CPU).
* **Core Rule Compliance:** Menjaga kesesuaian dengan rulebook Penyisihan COMPFEST 18 (Synchronous Snapshot, `docker-compose up`, CPU-only ONNX inference < 150 ms/frame).

---

## 2. Benchmark Jurnal Ilmiah Terkini (Hoang et al., 2026 & Prasetyo et al., 2024)
* **Referensi Utama 1 (DaFiF):** *Data in Brief 57 (2024) 111016* (Dataset DaFiF: 2.536 foto, 3 spesies, 11 hari penyimpanan es, organoleptik SNI 2729:2013).
* **Referensi Utama 2 (FFE):** *Deep feature optimization for enhanced fish freshness assessment* (Elsevier - Ecological Informatics 95, March 2026).

---

## 3. Bedah Temuan Kritis: Data Leakage & Domain Shift
Berdasarkan hasil eksperimen Kaggle, ditemukan temuan penting yang harus diangkat dalam proposal sebagai bukti integritas riset dan keunggulan engineering NusaQC:

1. **Random Split (99.48% Akurasi, 0.9939 F1):**
   * *Diagnosa:* Terjadi **Spatiotemporal Data Leakage**. Karena hanya ada 10 ikan per spesies yang dipotret berulang kali, Random Split menyebabkan foto dari ikan dan sesi yang sama tersebar di train dan test set. Model menghafal spesimen individu dan latar belakang laboratorium.
2. **Grouped Split by Day/Session (80.00% Akurasi, 0.7496 F1):**
   * *Diagnosa:* Performa **Generalisasi Realistis**. Ketika seluruh sesi dan hari diisolasi, akurasi menjadi 80.00% dengan Recall Grade C sebesar 82.86%. Nilai ini adalah estimasi jujur kesiapan model di lini produksi nyata.
3. **Secondary Test pada FFE (36.29% Akurasi):**
   * *Diagnosa:* **Modality & Field-of-View Mismatch**. DaFiF memotret tubuh dan kepala ikan secara utuh, sedangkan FFE adalah *macro close-up* pada mata. Fitur CNN yang mengekstraksi tubuh/insang/latar tidak dapat digeneralisasi langsung ke citra makro pupil tanpa model lokalisasi ROI khusus.

---

## 4. Keputusan Rekayasa Pipeline (Updated v2)

### A. All-in-One Fast In-Memory Script:
Pipeline digabung menjadi satu file terpadu yang mengeksekusi seluruh siklus secara otomatis dalam < 2 menit:
👉 [`models/model_1/01_model1_full_pipeline.py`](file:///d:/main/Documents/explore/compe/hackhathon/AIC/models/model_1/01_model1_full_pipeline.py)

### B. Fitur Baru yang Diimplementasikan:
1. **Class-Weighted CrossEntropyLoss:** Memberikan penalti berimbang untuk kelas minoritas Grade A (15.8%) dibanding Grade C (47.2%).
2. **Anti-Leakage Augmentations:** Menggunakan `RandomErasing` (Cutout), `ColorJitter`, dan `RandomResizedCrop` untuk memaksa model fokus pada morfologi ikan dan tidak bergantung pada latar belakang meja.
3. **Full EDA & Visualization Suite:**
   - `eda_dataset_distribution.png`: Distribusi kelas, spesies, degradasi hari ke hari, dan DaFiF vs FFE.
   - `eda_sample_images_by_grade.png`: Grid sampel visual per kelas.
   - `training_curves_comparison.png`: Kurva Loss, Accuracy, dan F1 train vs val per epoch.
   - `confusion_matrices_all.png`: Matriks kebingungan (raw counts & normalized %).
   - `sample_test_predictions_grid.png`: Visualisasi prediksi benar vs misklasifikasi.
4. **Fixed ONNX Export (Opset 17) & INT8 Dynamic Quantization:** Menghilangkan error konversi PyTorch Dynamo dan `ShapeInferenceError` pada ONNX Runtime.
5. **CPU Latency Benchmarking:** Pengujian latensi CPU (ms/frame) dan FPS untuk simulasi edge device Raspberry Pi 5.

---

## 5. Ringkasan Kuantitatif untuk Proposal

| Metrik Evaluasi | Random Split (Baseline) | Grouped Split (Honest) | Secondary FFE Test |
| :--- | :--- | :--- | :--- |
| **Akurasi Global** | 99.48% | 80.00% | 36.29% |
| **Macro F1-Score** | 0.9939 | 0.7496 | 0.2966 |
| **Recall Grade C (Safety Critical)** | 1.0000 | 0.8286 | 0.0256 |
| **Status Interpretasi** | *Artificially Inflated (Leakage)* | *Production Benchmark* | *Domain Mismatch* |

*Dokumen ini diperbarui sebagai acuan resmi engineering Model 1 NusaQC COMPFEST 18.*
