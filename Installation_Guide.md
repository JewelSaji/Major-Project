# TRANCE Framework: Installation & Setup Guide

This guide details the prerequisites and step-by-step instructions for deploying the TRANCE framework on a new system.

## 1. System Prerequisites

### Hardware Requirements
- **RAM**: Minimum 32GB (64GB recommended) to process the MIMIC-IV tabular data chunks efficiently.
- **GPU (Highly Recommended)**: An NVIDIA GPU (e.g., T4, RTX 3090, or A100) with at least 16GB VRAM. While the system can run on a CPU, generating text embeddings for hundreds of thousands of clinical notes using Transformer models (`ClinicalT5` or `Bio_ClinicalBERT`) will be prohibitively slow without a GPU.

### Software Requirements
- **OS**: Linux or Windows
- **Python**: 3.10 or higher
- **Data Access**: You must have an approved researcher credential on [PhysioNet](https://physionet.org/) to download the MIMIC datasets.

## 2. Required Datasets

Before running the pipeline, you need to download the following datasets from PhysioNet and extract them to your local storage:
1. **MIMIC-IV (v3.1)**: Core tabular EHR data (admissions, icustays, chartevents, labevents, etc.)
2. **MIMIC-IV-Note (v2.2)**: Unstructured clinical text (discharge summaries, radiology reports)
3. *(Optional)* MIMIC-IV-EXT BHC: Supplementary clinical notes dataset if used in extended extraction.

## 3. Installation Steps

### Step 1: Clone the Repository & Setup Virtual Environment
Open your terminal and create a fresh Python virtual environment:
```bash
git clone <your-repository-url> readmission-ai
cd readmission-ai
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install PyTorch (GPU Support)
Since text embeddings heavily rely on PyTorch, install the GPU-accelerated version *first* by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/) that match your system's CUDA version.
For example, for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Core Dependencies
Install the required packages using the provided requirements file:
```bash
pip install -r requirements.txt
```

### Step 4: Install Optional Extensions
The training script (`03_train.py`) uses a meta-learner ensemble that benefits significantly from `xgboost`, `catboost`, and class-balancing tools. Install these for maximum performance:
```bash
pip install xgboost catboost imbalanced-learn
```

## 4. Configuration

Before running any scripts, you must configure the dataset paths. 
Open `src/config.py` in your code editor and update the following variables to point to your unzipped downloaded MIMIC datasets:

```python
# src/config.py
MIMIC_IV_DIR = r"C:\path\to\physionet.org\files\mimiciv\3.1"
MIMIC_NOTE_DIR = r"C:\path\to\physionet.org\files\mimic-iv-note\2.2"
```

*Note: Ensure you use raw strings (`r"..."`) on Windows to prevent path escape character issues.*

## 5. Running the Pipeline

The framework is designed to run sequentially. Execute the scripts from the root directory in this exact order:

1. **Extract Tabular Features:** (Takes a while to process all CSVs)
   ```bash
   python src/01_extract.py
   ```
2. **Select & Prune Features:**
   ```bash
   python src/01b_select_features.py
   ```
3. **Generate Text Embeddings:** (GPU heavily utilized here)
   ```bash
   python src/02_embed.py
   ```
4. **Train Models:** (LightGBM + XGBoost + Meta-Learner + Calibration)
   ```bash
   python src/03_train.py
   ```
5. **Analyze & Validate (Optional):**
   ```bash
   python src/04_diagnose.py
   python src/05_analyze.py
   python src/06_visualize.py
   ```

## 6. Serving Predictions

Once `03_train.py` completes, the serialized model is saved to `models/trance_framework.pkl`. You can then interact with the model in two ways:

**Option A: Interactive CLI Terminal**
```bash
python src/08_predict.py
```

**Option B: REST API Server**
```bash
uvicorn src.07_api:app --host 0.0.0.0 --port 8000
```
This will start a FastAPI server. You can view the OpenAPI documentation and test the endpoints directly by navigating to `http://localhost:8000/docs` in your web browser.
