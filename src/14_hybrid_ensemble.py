"""
TRANCE-Hybrid: Ensemble of LightGBM and TRANCE-Gate
==================================================
This script combines the predictions of:
1. Base Ensemble (LightGBM/XGBoost calibrated probabilities)
2. TRANCE-Gate (Neural Gated Fusion calibrated probabilities)

It performs a weighted average of the calibrated probabilities to achieve
potentially better AUROC and more robust calibration.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import (
        MAIN_MODEL_PKL, GATE_MODEL_PKL, RESULTS_DIR, FIGURES_DIR,
        RANDOM_STATE
    )
except ImportError:
    from config import (
        MAIN_MODEL_PKL, GATE_MODEL_PKL, RESULTS_DIR, FIGURES_DIR,
        RANDOM_STATE
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, total = 0.0, len(labels)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)

def run_hybrid_ensemble():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info("Loading Base Ensemble results from %s", MAIN_MODEL_PKL)
    if not os.path.exists(MAIN_MODEL_PKL):
        logger.error("Base model file not found!")
        return
    base_bundle = joblib.load(MAIN_MODEL_PKL)
    
    logger.info("Loading TRANCE-Gate results from %s", GATE_MODEL_PKL)
    if not os.path.exists(GATE_MODEL_PKL):
        logger.error("TRANCE-Gate model file not found!")
        return
    gate_bundle = joblib.load(GATE_MODEL_PKL)
    
    # 1. Extract data
    base_ids = base_bundle.get("test_hadm_ids")
    base_probs = base_bundle.get("test_probs_cal")
    base_labels = base_bundle.get("test_labels")
    
    gate_ids = gate_bundle.get("test_hadm_ids")
    gate_probs = gate_bundle.get("test_probs_cal")
    gate_labels = gate_bundle.get("test_labels")
    
    if base_ids is None or gate_ids is None:
        logger.error("Test IDs missing in model bundles.")
        return
        
    logger.info("Base test size: %d | Gate test size: %d", len(base_ids), len(gate_ids))
    
    # 2. Align predictions
    # Create DataFrames for easy merging
    df_base = pd.DataFrame({'hadm_id': base_ids, 'p_base': base_probs, 'y_base': base_labels})
    df_gate = pd.DataFrame({'hadm_id': gate_ids, 'p_gate': gate_probs, 'y_gate': gate_labels})
    
    df_hybrid = df_base.merge(df_gate, on='hadm_id', how='inner')
    logger.info("Aligned test size: %d", len(df_hybrid))
    
    if len(df_hybrid) == 0:
        logger.error("No overlapping hadm_ids found! Check if models used the same split.")
        return

    # Check for label consistency
    if not np.array_equal(df_hybrid['y_base'].values, df_hybrid['y_gate'].values):
        logger.warning("Labels are not identical in aligned set! This is unexpected.")
    
    y_true = df_hybrid['y_base'].values
    p_base = df_hybrid['p_base'].values
    p_gate = df_hybrid['p_gate'].values
    
    # 3. Hybrid Ensembling (Simple Average as starting point)
    # We could optimize weights on validation, but simple average is often robust.
    # W=0.5 / 0.5
    p_hybrid = 0.5 * p_base + 0.5 * p_gate
    
    # 4. Metrics
    metrics = {
        "Base-Ensemble": {
            "auroc": roc_auc_score(y_true, p_base),
            "auprc": average_precision_score(y_true, p_base),
            "ece": compute_ece(p_base, y_true),
            "brier": brier_score_loss(y_true, p_base)
        },
        "TRANCE-Gate": {
            "auroc": roc_auc_score(y_true, p_gate),
            "auprc": average_precision_score(y_true, p_gate),
            "ece": compute_ece(p_gate, y_true),
            "brier": brier_score_loss(y_true, p_gate)
        },
        "TRANCE-Hybrid": {
            "auroc": roc_auc_score(y_true, p_hybrid),
            "auprc": average_precision_score(y_true, p_hybrid),
            "ece": compute_ece(p_hybrid, y_true),
            "brier": brier_score_loss(y_true, p_hybrid)
        }
    }
    
    logger.info("="*50)
    logger.info("HYBRID ENSEMBLE RESULTS")
    for model, m in metrics.items():
        logger.info(f"{model:20s} | AUROC: {m['auroc']:.4f} | AUPRC: {m['auprc']:.4f} | ECE: {m['ece']:.4f}")
    logger.info("="*50)
    
    # 5. Save results
    report_path = os.path.join(RESULTS_DIR, "hybrid_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Hybrid report saved -> %s", report_path)
    
    # Optionally save hybrid predictions
    df_hybrid['p_hybrid'] = p_hybrid
    preds_path = os.path.join(RESULTS_DIR, "hybrid_predictions.csv")
    df_hybrid.to_csv(preds_path, index=False)
    logger.info("Hybrid predictions saved -> %s", preds_path)

if __name__ == "__main__":
    run_hybrid_ensemble()
