# TRANCE: Final Project Results and Metrics

This document summarizes the final outcomes, performance metrics, and technical contributions of the TRANCE (Text-guided Readmission prediction with Adaptive Neural Context-aware gating and Ensemble learning) project.

## 1. Project Overview: What We Did

We developed an imaging-free, clinically deployable system for 30-day hospital readmission prediction using the MIMIC-IV dataset. The key activities included:

- **Multimodal Feature Engineering**: Extracted 350+ structured EHR features (vitals, labs, medications, comorbidities) and generated 256-dimensional clinical text embeddings using **ClinicalT5**.
- **Base Model Development**: Optimized a high-performance ensemble of **LightGBM** and **XGBoost** using Optuna-based hyperparameter tuning.
- **Novel Gating Architecture**: Implemented **TRANCE-Gate**, a PyTorch-based neural network where clinical text embeddings dynamically amplify or suppress structured features via a learned gating mechanism.
- **Clinical Calibration**: Applied **Isotonic Regression** to all models to ensure that predicted probabilities correspond directly to real-world risk, a prerequisite for clinical utility.
- **Multi-Dimensional Evaluation**:
    - **Interpretability**: Verified that the model learns to suppress contextually expected abnormalities (e.g., low hemoglobin in chronic anemia patients).
    - **Fairness**: Evaluated performance across race, gender, and age subgroups to ensure equitable care.
    - **Early Warning**: Tested the model's ability to predict readmission as early as Day 1 and Day 2 of hospitalization.
    - **Temporal Stability**: Confirmed model robustness across 15 years of data (2008–2022).

## 2. Core Performance Metrics

The table below compares the Base Ensemble (LightGBM) with the TRANCE-Gate architecture.

| Metric | Base Ensemble (LightGBM) | TRANCE-Gate (Gated Fusion) | TRANCE-Hybrid (Ensemble) |
| :--- | :---: | :---: | :---: |
| **AUROC (Overall)** | 0.7705 | 0.7683 | **0.7738** |
| **AUPRC** | 0.4708 | 0.4679 | **0.4838** |
| **Expected Calibration Error (ECE)** | 0.0036 | 0.0058 | 0.0061 |
| **Brier Score** | 0.1245 | 0.1249 | 0.1241 |

> [!TIP]
> **TRANCE-Hybrid** achieves the "best of both worlds" by combining the high discriminative power of LightGBM with the context-aware interpretability of the Gated Fusion model.

> [!NOTE]
> While TRANCE-Gate's discriminative performance (AUROC) is comparable to the base ensemble, its primary value lies in its context-aware interpretability and its ability to learn patient-specific feature weighting from clinical notes.

## 3. Analysis Results

### Early Warning Performance
Our model demonstrates remarkable stability early in the patient stay, allowing for proactive discharge planning.

| Admission Day | AUROC |
| :--- | :---: |
| **Day 1** | 0.7708 |
| **Day 2** | 0.7710 |
| **Full Stay** | 0.7707 |

### Temporal Stability (2008–2022)
The model maintains consistent performance across nearly 15 years of clinical data, with slight variations likely due to changing clinical practices.

| Year Group | Admissions (n) | AUROC (TRANCE-Gate) |
| :--- | :---: | :---: |
| 2008-2010 | 34,549 | 0.7641 |
| 2011-2013 | 17,556 | 0.7677 |
| 2014-2016 | 13,478 | 0.7699 |
| 2017-2019 | 10,287 | 0.7726 |
| 2020-2022 | 6,371 | 0.7591 |

### Fairness and Demographic Equity
We evaluated the model across several protected subgroups.

- **Gender**: Performance is slightly higher for Female patients (0.7739) compared to Male patients (0.7616).
- **Age**: The model performs exceptionally well for younger patients (AUROC 0.84 for age <40) but faces greater challenges with elderly patients (AUROC 0.68 for age 75-84), reflecting the inherent complexity of geriatric readmissions.
- **Race**: Performance remains stable across quartiles, with AUROC ranging from 0.77 to 0.79.

## 4. Conclusion
The TRANCE project successfully demonstrates that clinical text can be used as a dynamic controller for structured EHR data. By providing calibrated, fair, and early predictions, the system moves beyond raw benchmarks toward a truly deployable clinical decision support tool.
