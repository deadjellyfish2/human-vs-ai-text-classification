# Human vs AI Text Classification

This repository/notebook contains a text classification experiment to distinguish between **Human-written text** and **AI-generated text** using three main models:

1. **IndoBERT** (`indobenchmark/indobert-base-p2`)
2. **XLM-RoBERTa** (`xlm-roberta-base`)
3. **BiLSTM + FastText Indonesian pretrained embedding**

Main notebook: `[AOL]nlp_ai_classification.ipynb`

---

## Process Overview

This code runs an end-to-end NLP experiment pipeline, starting from dataset loading, data validation, exploratory data analysis, model training, evaluation, and saving experiment outputs to Google Drive.

The main steps are:

1. **Environment setup**
   - Installs the required libraries.
   - Mounts Google Drive.
   - Imports Python libraries such as PyTorch, Transformers, scikit-learn, pandas, matplotlib, seaborn, and wordcloud.

2. **Experiment configuration**
   - Sets the random seed for better reproducibility.
   - Defines the text and label columns.
   - Sets the output directory in Google Drive.
   - Defines model configurations such as max length, batch size, learning rate, epochs, dropout, and weight decay.

3. **Dataset upload and validation**
   - The dataset is uploaded through `files.upload()` in Google Colab.
   - The dataset must contain at least two columns:
     - `teks`
     - `label`
   - Labels are normalized into:
     - `0` = Human
     - `1` = AI
   - Missing values and exact duplicate texts are removed.

4. **Exploratory Data Analysis (EDA)**
   - Calculates word count and character count.
   - Visualizes label distribution.
   - Visualizes text length distribution.
   - Compares text length between Human and AI classes.
   - Shows the most frequent words.
   - Generates wordclouds for all data, Human texts, and AI texts.

5. **Dataset splitting**
   - The dataset is split into:
     - 70% training
     - 15% validation
     - 15% test
   - The split is stratified by label.
   - The split files are saved to Google Drive.

6. **Leakage check**
   - The code checks whether exact duplicate texts appear across the train, validation, and test sets.
   - This is done to reduce the risk of data leakage.

7. **Class imbalance handling**
   - The code checks the class distribution.
   - If the class ratio gap exceeds the defined threshold, class weights are used in the loss function.

8. **Transformer model training**
   - IndoBERT and XLM-RoBERTa are trained using the following architecture:
     - pretrained Transformer encoder
     - first-token representation (`[CLS]` or `<s>`)
     - dropout
     - linear classifier
   - The models are evaluated on the validation set after each epoch.
   - The best model is selected based on validation F1-score.

9. **BiLSTM + FastText training**
   - Text is cleaned and tokenized using simple whitespace-based tokenization.
   - The vocabulary is built only from the training set.
   - Indonesian FastText pretrained embedding is downloaded and loaded.
   - BiLSTM reads the text sequence in both forward and backward directions.
   - The final forward and backward hidden states are concatenated and passed to the classifier.
   - Early stopping is applied based on validation F1-score.

10. **Model evaluation**
    - All models are evaluated using the same metrics:
      - Accuracy
      - Precision
      - Recall
      - F1-score
      - ROC-AUC
      - Confusion matrix
      - Classification report
      - Confidence score
    - Predictions and confidence scores are saved as CSV files.

11. **Model comparison**
    - Compares the performance of IndoBERT, XLM-RoBERTa, and BiLSTM + FastText.
    - Generates metric comparison plots.
    - Combines predictions from all models into a single file.

12. **Feature representation visualization**
    - Extracts feature representations from:
      - IndoBERT CLS embedding
      - XLM-RoBERTa `<s>` embedding
      - BiLSTM final hidden state
    - Visualizes the features using 2D PCA.

13. **Saving outputs**
    - All outputs are saved to Google Drive, including:
      - EDA results
      - dataset splits
      - model artifacts
      - training history
      - metrics JSON files
      - classification reports
      - confusion matrices
      - prediction CSV files
      - model comparison charts
      - PCA visualizations
      - zipped experiment results

---

## Output Structure

By default, outputs are saved to:

```text
/content/drive/MyDrive/Deep-Learning/AOL_II/Results3/HumanVsAI_Text-Classification
```

Main output folder structure:

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

The final results are also compressed into:

```text
Classification-Result_HumanVsAI.zip
```

---

## Dataset Format

The input dataset must be a CSV file with at least the following columns:

```csv
teks,label
"Example text written by a human",0
"Example text generated by AI",1
```

Supported label values:

| Label | Meaning |
|---|---|
| `0` | Human |
| `1` | AI |
| `human` / `manusia` | Human |
| `ai` / `machine` / `generated` | AI |

---

## How to Run the Code

### 1. Open the notebook in Google Colab

Upload or open the file:

```text
[AOL]nlp_ai_classification.ipynb
```

GPU runtime is recommended:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

---

### 2. Install the required libraries

Run the first cell:

```python
!pip install -q transformers sentencepiece accelerate scikit-learn matplotlib wordcloud h5py onnx onnxruntime
```

---

### 3. Mount Google Drive

Run the following cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Grant Google Drive access so the experiment outputs can be saved.

---

### 4. Run imports and configuration

Run the import and configuration cells.

Make sure these variables match your dataset:

```python
TEXT_COL = "teks"
LABEL_COL = "label"
```

If your dataset uses different column names, change `TEXT_COL` and `LABEL_COL` accordingly.

---

### 5. Upload the CSV dataset

When this section runs:

```python
uploaded = files.upload()
```

select the CSV dataset file.

---

### 6. Run EDA and dataset splitting

Run the cells in the EDA and split sections.

This step will:
- remove missing values and exact duplicates,
- calculate text statistics,
- generate visualizations,
- split the data into train/validation/test sets,
- run leakage checks,
- calculate class weights if needed.

---

### 7. Train IndoBERT

Run:

```python
indobert_model, indobert_tokenizer, indobert_metrics, indobert_pred_df = train_transformer_model(
    model_name=INDOBERT_MODEL_NAME,
    model_short_name="IndoBERT"
)
```

---

### 8. Train XLM-RoBERTa

Run:

```python
xlmr_model, xlmr_tokenizer, xlmr_metrics, xlmr_pred_df = train_transformer_model(
    model_name=XLMR_MODEL_NAME,
    model_short_name="XLM_RoBERTa"
)
```

---

### 9. Train BiLSTM + FastText

Run the BiLSTM section in the correct order:

1. Download the Indonesian FastText embedding.
2. Build the vocabulary from the training set.
3. Define the FastText embedding loading function.
4. Load the embedding matrix.
5. Define the dataset and BiLSTM model.
6. Run BiLSTM training.

Important note: make sure `vocab` and `load_pretrained_embedding_matrix()` are already defined before calling:

```python
embedding_matrix, EMBED_DIM, embedding_summary = load_pretrained_embedding_matrix(
    embedding_path=PRETRAINED_EMBEDDING_PATH,
    vocab=vocab
)
```

If you encounter `NameError: name 'vocab' is not defined` or `NameError: name 'load_pretrained_embedding_matrix' is not defined`, run the vocabulary creation cell and the embedding loading function definition cell first.

Also add these imports if they are not already included:

```python
import gzip
import copy
```

---

### 10. Run model comparison

Run the comparison section.

This section generates:
- model comparison metric table,
- comparison charts for accuracy, precision, recall, F1-score, and ROC-AUC,
- confidence summary,
- confidence score distribution plots.

---

### 11. Run feature representation visualization

Run the feature representation section.

This section extracts feature representations from the three models and visualizes them using 2D PCA.

---

### 12. Save the final results

Run the final cell to create a zipped archive of the experiment outputs:

```python
shutil.make_archive(
    base_name=str(zip_base_path),
    format="zip",
    root_dir=OUTPUT_ROOT
)
```

The zip file will be saved to Google Drive.

---

## Models Used

### IndoBERT

IndoBERT is used as an Indonesian Transformer-based model. It uses WordPiece tokenization and the `[CLS]` token representation as the main feature for classification.

### XLM-RoBERTa

XLM-RoBERTa is used as a multilingual Transformer baseline. It uses SentencePiece tokenization and the `<s>` token representation as the main feature for classification.

### BiLSTM + FastText

BiLSTM + FastText is used as a classical deep learning baseline. It uses Indonesian FastText pretrained word vectors as embeddings and processes word sequences using a bidirectional LSTM.

---

## Main Configuration

| Parameter | IndoBERT | XLM-RoBERTa | BiLSTM + FastText |
|---|---:|---:|---:|
| Max length | 192 | 192 | 192 |
| Batch size | 16 | 16 | 32 |
| Epochs | 3 | 3 | 15 |
| Learning rate | 2e-5 | 2e-5 | 2e-4 |
| Dropout | 0.5 on classifier head | 0.5 on classifier head | 0.5 |
| Loss | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss |
| Optimizer | AdamW | AdamW | AdamW |

---

## Experiment Notes

- This notebook does not perform text data augmentation.
- Model comparison is performed using the same dataset, split, and evaluation metrics.
- BiLSTM + FastText is used as a baseline to compare the static embedding + recurrent neural network approach against Transformer-based contextual embedding models.
- The BiLSTM vocabulary is built only from the training set to avoid data leakage.
- The FastText embedding is stored only in the Colab runtime at `/content/embeddings/cc.id.300.vec.gz`.
- ONNX export is optional. If it fails, the error is saved in `onnx_export_error.txt`.

---

## Important Output Files

| File | Description |
|---|---|
| `model_comparison_metrics.csv` | Metric comparison across all models |
| `all_models_predictions_with_confidence.csv` | Predictions and confidence scores from all models |
| `confidence_summary.csv` | Confidence score summary per model |
| `training_history.csv` | Training history for each model |
| `test_metrics.json` | Test set evaluation metrics |
| `classification_report.txt` | Classification report |
| `confusion_matrix.png` | Confusion matrix |
| `fairness_experiment_summary.json` | Fairness summary of the experiment |
| `Classification-Result_HumanVsAI.zip` | Archive containing all experiment outputs |

---

## Research Objective

This notebook is used to evaluate and compare the effectiveness of Transformer-based models and a classical deep learning baseline for detecting **Human-written text** and **AI-generated text**. The experiment helps determine whether contextual embedding models such as IndoBERT and XLM-RoBERTa are more effective than BiLSTM with pretrained static FastText embeddings.
