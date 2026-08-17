# AI Context Map — Project NusaQC (COMPFEST 18 AIC)

File ini dibuat khusus sebagai acuan cepat (**Context Map**) untuk agen AI dan developer dalam menavigasi seluruh dokumen proyek **NusaQC** (Quality Control & Freshness Assessment berbasis AI).

---

## 📌 Quick Summary Proyek
- **Nama Utama Proyek**: **NusaQC** (Quality Control & Visual Inspection System)
  - *Catatan Naming*: Document lama/ideation kadang menyebut codename internal *NusaCatch*. Nama resmi submisi proposal dan sistem adalah **NusaQC**.
- **Kompetisi**: COMPFEST 18 - Artificial Intelligence Competition (AIC) · Smart Manufacturing Track
- **Fokus Utama**: Sistem inspeksi mutu dan pemilah otomatis berbasis Computer Vision & IoT Conveyor Control untuk rantai pasok industri perikanan ekspor Indonesia (UPI).
- **Arsitektur Dual AI Model**:
  1. **Model 1 (Kesegaran / Freshness)**: Deep feature extraction (ResNet50 / MobileNetV3) + Classifier (Eye & Gill dataset: FFE & DaFiF) -> Output Grade A, B, C.
  2. **Model 2 (Deteksi Defek / Defect Detection)**: YOLOv8 Object Detection dengan pseudo-labeling otomatis (3.200+ citra dari HF, Kaggle & Roboflow) -> Bounding Box 4 kelas defek permukaan (BGD, BRD, FDS, PD).

---

## 📂 Peta Struktur Direktori & Rujukan Konteks

### 1. 📄 `proposal/` (Dokumen Proposal Hackathon)
Gunakan direktori ini ketika AI perlu membaca, merevisi, atau mereferensikan isi proposal resmi:
- [`FIX_NusaQC_Proposal_AIC_COMPFEST18_2026_v3.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/proposal/FIX_NusaQC_Proposal_AIC_COMPFEST18_2026_v3.md) : **⭐ Proposal Utama Active (v3)** — Versi proposal paling mutakhir dengan statistik KKP/BPS/FDA terverifikasi & rancangan snapshot vs continuous conveyor.
- [`Rencana_Struktur_Proposal.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/proposal/Rencana_Struktur_Proposal.md) : Breakdown outline dan struktur bab/bagian proposal.
- [`Draft_Proposal_Inovasi_v1.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/proposal/Draft_Proposal_Inovasi_v1.md) & `.pdf` : Draft awal proposal inovasi (v1) *(Historical reference)*.

### 2. 💡 `ideation/` (Analisis Ide, Spesifikasi & Pembagian Tugas)
Gunakan direktori ini ketika AI perlu memahami latar belakang ide, kritik, spesifikasi teknis, atau pembagian kerja tim:
- [`Solidification_Main_Idea.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/Solidification_Main_Idea.md) : Pembekuan ide utama & arsitektur solusi.
- [`Laporan_Analisis_Bedah_Ide_NusaQC.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/Laporan_Analisis_Bedah_Ide_NusaQC.md) : Bedah problem statement & pemetaan solusi.
- [`Laporan_Evaluasi_Kritis_NusaQC.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/Laporan_Evaluasi_Kritis_NusaQC.md) : Analisis kelemahan ide, risiko & mitigasi.
- [`NusaCatch_Specs.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/NusaCatch_Specs.md) : Spesifikasi fitur & arsitektur sistem.
- [`MASTER_CONTEXT_DAN_REVISI_TIM_NUSAQC.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/MASTER_CONTEXT_DAN_REVISI_TIM_NUSAQC.md) : Master Context & Lembar Kerja Revisi Tim NusaQC.
- [`Perbaikan_Ide_Rayka.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/Perbaikan_Ide_Rayka.md) : Catatan perbaikan ide spesifik.
- [`Rekomendasi_Scope_Spesies_NusaQC.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/Rekomendasi_Scope_Spesies_NusaQC.md) : Penentuan batasan jenis/spesies ikan.
- [`Pembagian_Jobdesk.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/ideation/Pembagian_Jobdesk.md) : Pembagian tugas anggota tim.

### 3. 🔍 `research/` (Guidebook, Analisis Pemenang & Sitasi Data)
Gunakan direktori ini ketika AI perlu mencocokkan aturan lomba atau benchmarking dengan pemenang tahun lalu:
- [`guidebook.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/research/guidebook.md) : Guidebook resmi kompetisi COMPFEST 18.
- [`past_winners_analysis.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/research/past_winners_analysis.md) : Rangkuman & pola keberhasilan pemenang kompetisi terdahulu.
- [`Verifikasi_Sitasi_dan_Data_NusaQC.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/research/Verifikasi_Sitasi_dan_Data_NusaQC.md) : Verifikasi sitasi ilmiah & validasi data.
- [`PROMPT_KONSULTASI_AI.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/research/PROMPT_KONSULTASI_AI.md) : Collection prompt yang digunakan untuk konsultasi AI.
- `past_winners/` : Transkrip/berkas karya pemenang lalu ([ADA_SPARTANS.txt](file:///D:/main/Documents/explore/compe/hackhathon/AIC/research/past_winners/ADA_SPARTANS.txt), [Mechaminds.txt](file:///D:/main/Documents/explore/compe/hackhathon/AIC/research/past_winners/Mechaminds.txt), [tunarasa.txt](file:///D:/main/Documents/explore/compe/hackhathon/AIC/research/past_winners/tunarasa.txt)).

### 4. 📚 `docs/` (Paper Referensi & Spesifikasi Dataset)
Gunakan direktori ini untuk mengakses studi literatur ilmiah dan spesifikasi teknis dataset:
- `md/` : Ringkasan paper & spesifikasi:
  - [`Deep_feature_optimization_paper.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/docs/md/Deep_feature_optimization_paper.md) : Ringkasan paper utama optimasi fitur deep learning.
  - [`dafif.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/docs/md/dafif.md) : Ringkasan paper/dataset DaFiF.
  - [`datasets.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/docs/md/datasets.md) : Ringkasan analisis dataset.
  - [`spesifikasi_dataset_dan_scope_ai.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/docs/md/spesifikasi_dataset_dan_scope_ai.md) : Scope AI & pemetaannya terhadap dataset.
- `pdf/` : File asli PDF paper ilmiah.

### 5. 🤖 `models/` (Kode Machine Learning, Pipeline & Dataset)
Gunakan direktori ini untuk pengerjaan teknis ML, training, dan visualisasi:
- `model_1/` :
  - [`01_model1_full_pipeline.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_1/01_model1_full_pipeline.py) : Script lengkap training, ekstraksi fitur & evaluasi Model 1 (Eye & Gill).
  - [`MODEL_1_PIPELINE_GUIDE.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_1/MODEL_1_PIPELINE_GUIDE.md) : Panduan menjalankan dan memahami pipeline Model 1.
  - [`REKAP_DISKUSI_DAN_REKAYASA_AI_MODEL1.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_1/REKAP_DISKUSI_DAN_REKAYASA_AI_MODEL1.md) : Rekap rekayasa fitur & eksperimen Model 1.
- `model_2/` :
  - [`01_prepare_model2_dataset.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/01_prepare_model2_dataset.py) : Script inisialisasi folder, harmonisasi label, & dataset split.
  - [`02_label_studio_config.xml`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/02_label_studio_config.xml) : Template XML Label Studio 4 kelas (BGD, BRD, FDS, PD).
  - [`03_model2_kaggle_pipeline.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/03_model2_kaggle_pipeline.py) & `.ipynb` : Pipeline Kaggle GPU (YOLOv8s, Evaluasi & ONNX Export).
  - [`04_model2_kaggle_pseudolabeling.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/04_model2_kaggle_pseudolabeling.py) & `.ipynb` : Auto Pseudo-Labeling pipeline (HF panda992 + Alaa Mahmoud + Roboflow -> 3.200+ citra).
  - [`05_label_studio_converter.py`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/05_label_studio_converter.py) : Converter anotasik Label Studio ke format YOLO.
  - [`ANNOTATION_GUIDE_MODEL2.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/ANNOTATION_GUIDE_MODEL2.md) : Panduan taksonomi visual 4 kelas defek permukaan ikan.
  - [`MODEL_2_PIPELINE_GUIDE.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/MODEL_2_PIPELINE_GUIDE.md) : Panduan eksekusi pipeline Model 2.
  - [`REKAP_DISKUSI_DAN_REKAYASA_AI_MODEL2.md`](file:///D:/main/Documents/explore/compe/hackhathon/AIC/models/model_2/REKAP_DISKUSI_DAN_REKAYASA_AI_MODEL2.md) : Rekap rekayasa teknis & eksperimen Model 2.
- `datasets/` (*Git Ignored*):
  - `model-1/` : Folder lokal untuk dataset DaFiF & FFE.
  - `model-2/` : Folder lokal untuk dataset roboflow-fish-disease, nusaqc_extended_pseudo_dataset, dll.

---

## ⚡ Panduan Pemanggilan Context untuk AI (Prompting Tip)
Ketika memberikan instruksi baru kepada AI, Anda cukup merujuk bagian ini:
- **Untuk penulisan/revisi proposal**: *"Rujuk proposal/FIX_NusaQC_Proposal_AIC_COMPFEST18_2026_v3.md dan guidebook di research/guidebook.md"*
- **Untuk pengembangan Model 1 (Freshness)**: *"Rujuk pipeline di models/model_1/01_model1_full_pipeline.py dan panduannya di models/model_1/MODEL_1_PIPELINE_GUIDE.md"*
- **Untuk pengembangan Model 2 (Defect Detection)**: *"Rujuk pipeline di models/model_2/03_model2_kaggle_pipeline.py dan models/model_2/MODEL_2_PIPELINE_GUIDE.md"*
- **Untuk validasi ide/fitur**: *"Cek ideation/Solidification_Main_Idea.md dan docs/md/spesifikasi_dataset_dan_scope_ai.md"*
