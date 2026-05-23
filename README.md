# Human vs AI Text Classification

Repository/notebook ini berisi eksperimen klasifikasi teks untuk membedakan teks **Human** dan **AI-generated text** menggunakan tiga model utama:

1. **IndoBERT** (`indobenchmark/indobert-base-p2`)
2. **XLM-RoBERTa** (`xlm-roberta-base`)
3. **BiLSTM + FastText Indonesian pretrained embedding**

Notebook utama: `[AOL]nlp_ai_classification.ipynb`

---

## Ringkasan Proses

Code ini menjalankan pipeline eksperimen NLP secara end-to-end, mulai dari membaca dataset, melakukan validasi data, eksplorasi data, training model, evaluasi, hingga menyimpan hasil eksperimen ke Google Drive.

Tahapan utama yang dilakukan:

1. **Setup environment**
   - Install library yang dibutuhkan.
   - Mount Google Drive.
   - Import library Python seperti PyTorch, Transformers, scikit-learn, pandas, matplotlib, seaborn, dan wordcloud.

2. **Konfigurasi eksperimen**
   - Menentukan seed agar eksperimen lebih reproducible.
   - Menentukan kolom teks dan label.
   - Mengatur direktori output di Google Drive.
   - Menentukan konfigurasi model, seperti max length, batch size, learning rate, epoch, dropout, dan weight decay.

3. **Upload dan validasi dataset**
   - Dataset diupload melalui `files.upload()` di Google Colab.
   - Dataset wajib memiliki minimal dua kolom:
     - `teks`
     - `label`
   - Label dinormalisasi menjadi:
     - `0` = Human
     - `1` = AI
   - Data kosong dan teks duplikat dihapus.

4. **Exploratory Data Analysis (EDA)**
   - Menghitung jumlah kata dan karakter.
   - Menampilkan distribusi label.
   - Menampilkan distribusi panjang teks.
   - Membandingkan panjang teks Human dan AI.
   - Menampilkan kata paling sering muncul.
   - Membuat wordcloud untuk seluruh data, Human, dan AI.

5. **Split dataset**
   - Dataset dibagi menjadi:
     - 70% training
     - 15% validation
     - 15% test
   - Split dilakukan secara stratified berdasarkan label.
   - Hasil split disimpan ke Google Drive.

6. **Leakage check**
   - Code mengecek apakah ada teks duplikat persis yang muncul di train, validation, dan test set.
   - Tujuannya untuk menghindari data leakage.

7. **Class imbalance handling**
   - Code mengecek distribusi kelas.
   - Jika perbedaan rasio kelas melewati threshold tertentu, class weight digunakan pada loss function.

8. **Training model Transformer**
   - IndoBERT dan XLM-RoBERTa dilatih menggunakan arsitektur:
     - pretrained Transformer encoder
     - representasi token pertama (`[CLS]` atau `<s>`)
     - dropout
     - linear classifier
   - Model dievaluasi pada validation set setiap epoch.
   - Model terbaik dipilih berdasarkan validation F1-score.

9. **Training BiLSTM + FastText**
   - Teks dibersihkan dan ditokenisasi secara sederhana.
   - Vocabulary dibuat hanya dari training set.
   - FastText Indonesian pretrained embedding diunduh dan dimuat.
   - BiLSTM membaca teks secara dua arah.
   - Hidden state forward dan backward digabungkan, lalu masuk ke classifier.
   - Early stopping digunakan berdasarkan validation F1-score.

10. **Evaluasi model**
    - Semua model dievaluasi menggunakan metrik yang sama:
      - Accuracy
      - Precision
      - Recall
      - F1-score
      - ROC-AUC
      - Confusion matrix
      - Classification report
      - Confidence score
    - Hasil prediksi dan confidence score disimpan dalam file CSV.

11. **Perbandingan model**
    - Membandingkan performa IndoBERT, XLM-RoBERTa, dan BiLSTM + FastText.
    - Membuat grafik perbandingan metrik.
    - Menggabungkan semua prediksi model ke dalam satu file.

12. **Visualisasi feature representation**
    - Mengambil representasi fitur dari:
      - CLS embedding IndoBERT
      - `<s>` embedding XLM-RoBERTa
      - final hidden state BiLSTM
    - Fitur divisualisasikan menggunakan PCA 2D.

13. **Penyimpanan hasil**
    - Semua hasil disimpan ke Google Drive, termasuk:
      - hasil EDA
      - split dataset
      - model artifacts
      - training history
      - metrics JSON
      - classification report
      - confusion matrix
      - prediction CSV
      - grafik perbandingan model
      - visualisasi PCA
      - file zip hasil eksperimen

---

## Struktur Output

Secara default, output disimpan ke:

```text
/content/drive/MyDrive/Deep-Learning/AOL_II/Results3/HumanVsAI_Text-Classification
```

Struktur folder utama:

```text
HumanVsAI_Text-Classification/
├── eda/
│   ├── label_distribution.png
│   ├── word_count_distribution.png
│   ├── char_count_distribution.png
│   ├── wordcloud_all.png
│   ├── wordcloud_human.png
│   └── wordcloud_ai.png
│
├── leakage_check/
│   └── exact_overlap_summary.json
│
├── model_artifacts/
│   ├── IndoBERT/
│   ├── XLM_RoBERTa/
│   └── BiLSTM_FastText/
│
├── model_comparison/
│   ├── model_comparison_metrics.csv
│   ├── all_models_predictions_with_confidence.csv
│   ├── confidence_summary.csv
│   └── comparison_*.png
│
├── feature_representation/
│   ├── indobert_pca.png
│   ├── xlmr_pca.png
│   └── bilstm_pca.png
│
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── class_weight_summary.json
└── fairness_experiment_summary.json
```

Hasil akhir juga dikompres menjadi:

```text
Classification-Result_HumanVsAI.zip
```

---

## Format Dataset

Dataset input harus berupa file CSV dengan minimal kolom berikut:

```csv
teks,label
"Contoh teks yang ditulis manusia",0
"Contoh teks yang dihasilkan AI",1
```

Label yang didukung:

| Label | Makna |
|---|---|
| `0` | Human |
| `1` | AI |
| `human` / `manusia` | Human |
| `ai` / `machine` / `generated` | AI |

---

## Cara Menjalankan Code

### 1. Buka notebook di Google Colab

Upload atau buka file:

```text
[AOL]nlp_ai_classification.ipynb
```

Disarankan menggunakan runtime GPU:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

---

### 2. Jalankan instalasi library

Jalankan cell pertama:

```python
!pip install -q transformers sentencepiece accelerate scikit-learn matplotlib wordcloud h5py onnx onnxruntime
```

---

### 3. Mount Google Drive

Jalankan cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Berikan izin akses Google Drive agar hasil eksperimen bisa disimpan.

---

### 4. Jalankan import dan konfigurasi

Jalankan cell import library dan konfigurasi.

Pastikan bagian ini sesuai dengan dataset:

```python
TEXT_COL = "teks"
LABEL_COL = "label"
```

Jika nama kolom dataset berbeda, ubah nilai `TEXT_COL` dan `LABEL_COL`.

---

### 5. Upload dataset CSV

Saat menjalankan bagian:

```python
uploaded = files.upload()
```

pilih file CSV dataset yang akan digunakan.

---

### 6. Jalankan EDA dan split dataset

Jalankan cell-cell pada bagian:

```text
EDA
TRAINING
```

Bagian ini akan:
- membersihkan data kosong dan duplikat,
- menghitung statistik teks,
- membuat visualisasi,
- membagi data menjadi train/validation/test,
- melakukan leakage check,
- menghitung class weight jika diperlukan.

---

### 7. Train IndoBERT

Jalankan bagian:

```python
indobert_model, indobert_tokenizer, indobert_metrics, indobert_pred_df = train_transformer_model(
    model_name=INDOBERT_MODEL_NAME,
    model_short_name="IndoBERT"
)
```

---

### 8. Train XLM-RoBERTa

Jalankan bagian:

```python
xlmr_model, xlmr_tokenizer, xlmr_metrics, xlmr_pred_df = train_transformer_model(
    model_name=XLMR_MODEL_NAME,
    model_short_name="XLM_RoBERTa"
)
```

---

### 9. Train BiLSTM + FastText

Jalankan bagian BiLSTM secara berurutan:

1. Download FastText Indonesian embedding.
2. Buat vocabulary dari training set.
3. Definisikan fungsi load FastText embedding.
4. Load embedding matrix.
5. Definisikan dataset dan model BiLSTM.
6. Jalankan training BiLSTM.

Catatan penting: pastikan `vocab` dan fungsi `load_pretrained_embedding_matrix()` sudah dibuat sebelum memanggil:

```python
embedding_matrix, EMBED_DIM, embedding_summary = load_pretrained_embedding_matrix(
    embedding_path=PRETRAINED_EMBEDDING_PATH,
    vocab=vocab
)
```

Jika muncul error `NameError: name 'vocab' is not defined` atau `NameError: name 'load_pretrained_embedding_matrix' is not defined`, jalankan dulu cell pembuatan vocabulary dan cell definisi fungsi load embedding.

Tambahkan juga import berikut jika belum ada:

```python
import gzip
import copy
```

---

### 10. Jalankan perbandingan model

Jalankan bagian:

```text
Comparison
```

Bagian ini akan membuat:
- tabel perbandingan metrik,
- grafik perbandingan accuracy, precision, recall, F1-score, dan ROC-AUC,
- confidence summary,
- distribusi confidence score.

---

### 11. Jalankan visualisasi feature representation

Jalankan bagian:

```text
Feature Representation
```

Bagian ini akan mengekstrak representasi fitur dari ketiga model dan memvisualisasikannya dengan PCA 2D.

---

### 12. Simpan hasil akhir

Jalankan cell terakhir untuk membuat file zip hasil eksperimen:

```python
shutil.make_archive(
    base_name=str(zip_base_path),
    format="zip",
    root_dir=OUTPUT_ROOT
)
```

File zip akan tersimpan di Google Drive.

---

## Model yang Digunakan

### IndoBERT

IndoBERT digunakan sebagai model Transformer berbasis Bahasa Indonesia. Model ini memakai tokenizer WordPiece dan representasi token `[CLS]` sebagai fitur utama untuk klasifikasi.

### XLM-RoBERTa

XLM-RoBERTa digunakan sebagai pembanding Transformer multilingual. Model ini memakai SentencePiece tokenizer dan representasi token `<s>` sebagai fitur utama untuk klasifikasi.

### BiLSTM + FastText

BiLSTM + FastText digunakan sebagai baseline deep learning klasik. Model ini memakai FastText Indonesian pretrained word vectors sebagai embedding, lalu memproses urutan kata menggunakan BiLSTM.

---

## Konfigurasi Utama

| Parameter | IndoBERT | XLM-RoBERTa | BiLSTM + FastText |
|---|---:|---:|---:|
| Max length | 192 | 192 | 192 |
| Batch size | 16 | 16 | 32 |
| Epoch | 3 | 3 | 15 |
| Learning rate | 2e-5 | 2e-5 | 2e-4 |
| Dropout | 0.5 pada classifier head | 0.5 pada classifier head | 0.5 |
| Loss | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| Optimizer | AdamW | AdamW | AdamW |

---

## Catatan Eksperimen

- Tidak ada proses augmentasi data teks pada notebook ini.
- Perbandingan model menggunakan dataset, split, dan metrik evaluasi yang sama.
- BiLSTM + FastText digunakan sebagai baseline untuk membandingkan pendekatan static embedding + recurrent neural network dengan Transformer berbasis contextual embedding.
- Vocabulary BiLSTM dibuat hanya dari training set agar tidak terjadi data leakage.
- FastText embedding disimpan hanya di runtime Colab pada path `/content/embeddings/cc.id.300.vec.gz`.
- ONNX export bersifat opsional; jika gagal, error akan disimpan pada file `onnx_export_error.txt`.

---

## Ringkasan Output Penting

| File | Isi |
|---|---|
| `model_comparison_metrics.csv` | Perbandingan metrik semua model |
| `all_models_predictions_with_confidence.csv` | Prediksi dan confidence score semua model |
| `confidence_summary.csv` | Ringkasan confidence score per model |
| `training_history.csv` | Riwayat training tiap model |
| `test_metrics.json` | Metrik evaluasi test set |
| `classification_report.txt` | Classification report |
| `confusion_matrix.png` | Confusion matrix |
| `fairness_experiment_summary.json` | Ringkasan fairness eksperimen |
| `Classification-Result_HumanVsAI.zip` | Arsip seluruh hasil eksperimen |

---

## Tujuan Penelitian

Notebook ini digunakan untuk mengevaluasi dan membandingkan efektivitas model Transformer dan baseline deep learning klasik dalam mendeteksi teks **Human** dan **AI-generated text**. Hasil eksperimen dapat digunakan untuk melihat apakah model berbasis contextual embedding seperti IndoBERT dan XLM-RoBERTa lebih efektif dibanding BiLSTM dengan pretrained static embedding FastText.
