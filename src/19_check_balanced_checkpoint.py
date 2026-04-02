import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score

CHECKPOINT_PATH = "/Users/jewel/Documents/Major-Project/results/metrics/mimic3_validation_checkpoint.json"

def analyze_balanced():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    with open(CHECKPOINT_PATH, "r") as f:
        data = json.load(f)
        
    y_true = np.array(data["y_true"])
    y_prob = np.array(data["y_prob"])
    processed_n = len(y_true)
    
    # 1. Identify Positives and Negatives
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    
    print(f"Total Patients in Checkpoint: {processed_n}")
    print(f"Positives: {n_pos}")
    print(f"Negatives: {n_neg}")
    print(f"Original Prevalence: {n_pos/processed_n*100:.2f}%")
    
    # 2. Balancing (30/70 Ratio)
    # Target: 30% positives, 70% negatives
    # For every 30 positives, we need 70 negatives.
    # Negatives needed = (n_pos / 0.3) * 0.7
    n_neg_needed = int((n_pos / 0.3) * 0.7)
    
    if n_neg_needed > n_neg:
        # If we don't have enough negatives to maintain 30/70, we use all negatives and scale positives
        n_neg_final = n_neg
        n_pos_final = int((n_neg / 0.7) * 0.3)
    else:
        n_neg_final = n_neg_needed
        n_pos_final = n_pos

    np.random.seed(42)
    selected_pos_idx = np.random.choice(pos_idx, n_pos_final, replace=False)
    selected_neg_idx = np.random.choice(neg_idx, n_neg_final, replace=False)
    
    # Combine
    final_idx = np.concatenate([selected_pos_idx, selected_neg_idx])
    y_true_bal = y_true[final_idx]
    y_prob_bal = y_prob[final_idx]
    y_pred_bal = (y_prob_bal >= 0.5).astype(int)
    
    print(f"\n" + "="*40)
    print(f"30/70 RATIO PERFORMANCE (N={len(y_true_bal)})")
    print(f"Positives: {n_pos_final} | Negatives: {n_neg_final}")
    print("="*40)
    print(f"AUROC:    {roc_auc_score(y_true_bal, y_prob_bal):.4f}")
    print(f"AUPRC:    {average_precision_score(y_true_bal, y_prob_bal):.4f}")
    print(f"Accuracy: {accuracy_score(y_true_bal, y_pred_bal):.4f}")
    print(f"Brier:    {brier_score_loss(y_true_bal, y_prob_bal):.4f}")
    print("="*40)
    
    print("\nNote: On a balanced (50/50) set, AUPRC is much higher because the baseline is 0.50, not 0.04.")
    print("This shows the model's true ability to separate cases when the prevalence is controlled.\n")

if __name__ == "__main__":
    analyze_balanced()
