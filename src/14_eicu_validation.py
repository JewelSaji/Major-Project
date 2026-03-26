"""
eICU External Validation Script
===============================
Applies the frozen MIMIC-IV 30-day readmission models to the eICU database.
No retraining is performed. Missing features are zero-filled.
"""
import os
import sys
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)

# ── Setup ─────────────────────────────────────────────────────────
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# eICU dataset path
EICU_DIR = os.path.join(project_dir, "EICU Dataset")

def load_eicu_data():
    """Load and preprocess eICU tables to construct proxy readmission labels and features."""
    logger.info("Loading eICU patient table...")
    patient = pd.read_csv(os.path.join(EICU_DIR, "patient.csv.gz"), low_memory=False)
    
    # EICU lacks absolute timestamps across hospital admissions.
    # We construct a proxy 30-day readmission label:
    # 1. Sort by uniquepid and hospitaladmitoffset
    patient = patient.sort_values(["uniquepid", "hospitaldischargeyear", "hospitaladmitoffset"])
    
    # We map 'patienthealthsystemstayid' as the unique hospital admission ('hadm_id')
    # and 'uniquepid' as 'subject_id'
    # Drop duplicates so we have one row per hospital stay
    hosp_stays = patient.drop_duplicates(subset=["patienthealthsystemstayid"])
    
    # Proxy readmission: if there is a subsequent admission for the same uniquepid
    hosp_stays["next_stay"] = hosp_stays.groupby("uniquepid")["patienthealthsystemstayid"].shift(-1)
    hosp_stays["readmit_30"] = hosp_stays["next_stay"].notnull().astype(int)
    
    # Remove in-hospital deaths from being considered for readmission baseline
    hosp_stays = hosp_stays[hosp_stays["hospitaldischargestatus"] != "Expired"]
    
    logger.info(f"Cohort size after basic filtering: {len(hosp_stays)}")
    return hosp_stays

def engineer_features(hosp_stays, required_features):
    """Map eICU columns to MIMIC-IV required features."""
    logger.info("Engineering features mapped to MIMIC-IV schema...")
    df = hosp_stays.copy()
    
    # Map Demographics
    # eICU age "> 89" mapped to 91
    df["anchor_age"] = pd.to_numeric(df["age"].replace("> 89", "91"), errors="coerce").fillna(65.0)
    df["gender"] = (df["gender"] == "Female").astype(int)  # Assuming 1=Female, 0=Male based on Mimic standards
    
    # Map Utilisation
    df["los_days"] = df["hospitaldischargeoffset"] / (24 * 60) # minutes to days
    df["los_days"] = df["los_days"].clip(lower=0.1)
    df["prev_admissions"] = df.groupby("uniquepid").cumcount()
    
    mapped_count = 0
    zeroed_features = []
    
    out_df = pd.DataFrame(index=df.index)
    
    for f in required_features:
        if f in df.columns:
            out_df[f] = df[f]
            mapped_count += 1
        elif f == "anchor_age":
            out_df[f] = df["anchor_age"]
            mapped_count += 1
        elif f == "gender":
            out_df[f] = df["gender"]
            mapped_count += 1
        elif f == "los_days":
            out_df[f] = df["los_days"]
            mapped_count += 1
        elif f == "prev_admissions":
            out_df[f] = df["prev_admissions"]
            mapped_count += 1
        else:
            out_df[f] = 0.0
            zeroed_features.append(f)
            
    logger.warning(f"Zeroed out {len(zeroed_features)} features that could not be mapped directly.")
    return out_df, df["readmit_30"], mapped_count, zeroed_features, df

def load_predict_module():
    """Load src.08_predict to access the frozen container."""
    import importlib.util
    path = os.path.join(project_dir, "src", "08_predict.py")
    spec = importlib.util.spec_from_file_location("predict_module", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["predict_module"] = mod
    spec.loader.exec_module(mod)
    return mod

def bootstrap_ci(y_true, y_prob, metric_func, n_bootstraps=1000, seed=42):
    """Compute 95% CI using bootstrap."""
    rng = np.random.RandomState(seed)
    scores = []
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_prob_np), len(y_prob_np))
        if len(np.unique(y_true_np[indices])) < 2:
            continue
        score = metric_func(y_true_np[indices], y_prob_np[indices])
        scores.append(score)
    if not scores:
        return 0.0, 0.0
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (mask.sum() / len(y_prob)) * np.abs(bin_acc - bin_conf)
    return ece

def main():
    logger.info("Starting eICU External Validation Pipeline")
    
    predict_mod = load_predict_module()
    mc = predict_mod.model_container
    if not mc.model_data:
        logger.error("MIMIC-IV model artifacts not found! Run training first.")
        sys.exit(1)
        
    required_features = mc.model_data["features"]
    
    hosp_stays = load_eicu_data()
    X, y, mapped_count, zeroed_features, raw_df = engineer_features(hosp_stays, required_features)
    
    logger.info("Running inference on frozen ensemble...")
    predict_probs = mc.predict_proba(X)
    
    # Main Metrics
    auroc = roc_auc_score(y, predict_probs)
    auroc_ci = bootstrap_ci(y, predict_probs, roc_auc_score)
    
    auprc = average_precision_score(y, predict_probs)
    auprc_ci = bootstrap_ci(y, predict_probs, average_precision_score)
    
    brier = brier_score_loss(y, predict_probs)
    brier_ci = bootstrap_ci(y, predict_probs, brier_score_loss)
    
    ece = compute_ece(y, predict_probs)
    ece_ci = bootstrap_ci(y, predict_probs, compute_ece)
    
    # Subgroups
    subgroup_metrics = {}
    
    # Gender
    for g_val, g_label in [(0, "Male"), (1, "Female")]:
        mask = raw_df["gender"] == g_val
        if mask.sum() > 50 and len(np.unique(y[mask])) > 1:
            sg_auroc = roc_auc_score(y[mask], predict_probs[mask])
            subgroup_metrics[f"Gender_{g_label}"] = float(sg_auroc)
            
    # Age
    age_groups = [(18, 45), (46, 65), (66, 80), (81, 120)]
    for low, high in age_groups:
        mask = (raw_df["anchor_age"] >= low) & (raw_df["anchor_age"] <= high)
        if mask.sum() > 50 and len(np.unique(y[mask])) > 1:
            sg_auroc = roc_auc_score(y[mask], predict_probs[mask])
            subgroup_metrics[f"Age_{low}_{high}"] = float(sg_auroc)
    
    logger.info(f"AUROC: {auroc:.4f} (95% CI: {auroc_ci[0]:.4f}-{auroc_ci[1]:.4f})")
    logger.info(f"AUPRC: {auprc:.4f} (95% CI: {auprc_ci[0]:.4f}-{auprc_ci[1]:.4f})")
    logger.info(f"Brier: {brier:.4f} (95% CI: {brier_ci[0]:.4f}-{brier_ci[1]:.4f})")
    logger.info(f"ECE:   {ece:.4f} (95% CI: {ece_ci[0]:.4f}-{ece_ci[1]:.4f})")
    
    # JSON Report
    report = {
        "cohort_size": int(len(X)),
        "readmission_rate": float(y.mean()),
        "mapped_features_count": int(mapped_count),
        "zeroed_features_count": int(len(zeroed_features)),
        "zeroed_features": zeroed_features,
        "metrics": {
            "AUROC": {"value": float(auroc), "ci_lower": float(auroc_ci[0]), "ci_upper": float(auroc_ci[1])},
            "AUPRC": {"value": float(auprc), "ci_lower": float(auprc_ci[0]), "ci_upper": float(auprc_ci[1])},
            "Brier": {"value": float(brier), "ci_lower": float(brier_ci[0]), "ci_upper": float(brier_ci[1])},
            "ECE": {"value": float(ece), "ci_lower": float(ece_ci[0]), "ci_upper": float(ece_ci[1])}
        },
        "subgroups_auroc": subgroup_metrics
    }
    
    os.makedirs(os.path.join(project_dir, "results", "metrics"), exist_ok=True)
    out_path = os.path.join(project_dir, "results", "metrics", "eicu_validation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Validation report saved to {out_path}")
    
if __name__ == "__main__":
    main()
