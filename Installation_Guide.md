# TRANCE Framework: Installation & Setup Guide

This guide details the prerequisites and step-by-step instructions for deploying the TRANCE framework on a new system.

---

## 1. System Prerequisites

### Hardware Requirements
- **RAM**: Minimum 32 GB (64 GB recommended) to process MIMIC-IV tabular data chunks efficiently.
- **GPU (Highly Recommended)**: An NVIDIA GPU (e.g., T4, RTX 3090, or A100) with at least 16 GB VRAM.  
  Without a GPU, generating text embeddings for hundreds of thousands of clinical notes will be prohibitively slow.
- **Storage**: ~100 GB free disk space for the raw MIMIC datasets and intermediate outputs.

### Software Requirements
- **OS**: Linux or Windows
- **Python**: 3.10 or higher
- **Data Access**: Approved researcher credentials on [PhysioNet](https://physionet.org/) are required to download the MIMIC datasets.

---

## 2. Required Datasets

Download the following datasets from PhysioNet and place them under a common `physionet.org/files/` directory:

| Dataset | Version | Contents |
|---|---|---|
| **MIMIC-IV** | v3.1 | Core tabular EHR data (admissions, icustays, labevents, etc.) |
| **MIMIC-IV-Note** | v2.2 | Unstructured clinical text (discharge summaries, radiology reports) |
| **MIMIC-IV-EXT BHC** *(optional)* | v1.2.0 | Supplementary labelled clinical notes for hospital course summarisation |

Expected directory layout after download:

```
physionet.org/
└── files/
    ├── mimiciv/3.1/
    ├── mimic-iv-note/2.2/
    └── mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0/
```

---

## 3. Installation Steps

### Step 1: Clone the Repository & Set Up a Virtual Environment

```bash
git clone <your-repository-url> readmission-ai
cd readmission-ai

python -m venv venv

# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 2: Install PyTorch (GPU Support)

Install the GPU-accelerated version of PyTorch **first** by following the
[official PyTorch instructions](https://pytorch.org/get-started/locally/) for your CUDA version.

Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Example for CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Core Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
shap>=0.42.0
torch>=2.0.0
transformers>=4.30.0
sentencepiece>=0.1.99
aiofiles>=23.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
scipy>=1.10.0
peft>=0.11.0
datasets>=2.18.0
accelerate>=0.30.0
```

### Step 4: Install Optional Extensions

The training script (`03_train.py`) benefits from `xgboost`, `catboost`, and class-balancing tools:

```bash
pip install xgboost catboost imbalanced-learn sentence-transformers
```

---

## 4. Configuration

Open `src/config.py` and verify the three MIMIC source paths point to your dataset location:

```python
# src/config.py  — Linux example (already set for this machine)
MIMIC_IV_DIR   = "/path/to/physionet.org/files/mimiciv/3.1"
MIMIC_NOTE_DIR = "/path/to/physionet.org/files/mimic-iv-note/2.2"
MIMIC_BHC_DIR  = "/path/to/physionet.org/files/mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0"

# Windows example (use raw strings to avoid escape issues)
MIMIC_IV_DIR   = r"C:\path\to\physionet.org\files\mimiciv\3.1"
MIMIC_NOTE_DIR = r"C:\path\to\physionet.org\files\mimic-iv-note\2.2"
```

All other output paths (`data/`, `models/`, `results/`, `figures/`) are derived automatically from the repository root and require no changes.

---

## 5. Running the Pipeline

### Option A — One-Click Automated Run *(recommended)*

```bash
python run_pipeline.py
```

This executes all batch scripts sequentially and prints a progress summary on completion.  
To also run the optional ClinicalT5 LoRA fine-tuning step before embedding:

```bash
RUN_CT5_FINETUNE=1 python run_pipeline.py   # Linux / macOS
set RUN_CT5_FINETUNE=1 && python run_pipeline.py  # Windows
```

### Option B — Run Scripts Individually

Execute scripts from the **repository root** in this order:

| Step | Script | Description |
|---|---|---|
| 1 | `python src/01_extract.py` | Extract & engineer tabular features from MIMIC-IV |
| 2 | `python src/01b_select_features.py` | SHAP-based feature pruning |
| 3 *(opt)* | `python src/02a_finetune_clinical_t5.py` | ClinicalT5 LoRA fine-tuning |
| 4 | `python src/02_embed.py` | Generate ClinicalT5 text embeddings (GPU-heavy) |
| 5 | `python src/03_train.py` | Train LightGBM + XGBoost ensemble with calibration |
| 6 | `python src/04_diagnose.py` | Embedding diagnostics & sanity checks |
| 7 | `python src/05_analyze.py` | SHAP interpretability analysis |
| 8 | `python src/06_visualize.py` | Journal-quality result visualisations |
| 9 | `python src/09_compare_models.py` | Cross-paper model benchmark comparison |

> **Note:** Steps 6–9 are optional validation/output steps and can be re-run independently without retraining.

---

## 6. Serving Predictions

Once `03_train.py` completes, the serialized model is saved to `models/trance_framework.pkl`.  
You can interact with the model in two ways:

### Option A — Interactive CLI Terminal

```bash
python src/08_predict.py
```

### Option B — REST API Server

```bash
python src/07_api.py
# or equivalently:
uvicorn src.07_api:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive OpenAPI documentation.

---

## 7. Output Files

| Location | Contents |
|---|---|
| `data/ultimate_features.csv` | Engineered tabular feature set |
| `data/embeddings.csv` | ClinicalT5 / SentenceTransformer embeddings |
| `models/trance_framework.pkl` | Serialised ensemble model |
| `models/feature_importance_report.csv` | SHAP feature rankings |
| `results/` | Prediction payloads, evaluation metrics |
| `figures/` | Plots and visualisation outputs |
