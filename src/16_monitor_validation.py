import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Path to the checkpoint file
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "metrics", "mimic3_validation_checkpoint.json")

def monitor():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint found yet at {CHECKPOINT_PATH}. Wait for the first 2,000 patients!")
        return

    try:
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
            
        y_true = np.array(data["y_true"])
        y_prob = np.array(data["y_prob"])
        genders = np.array(data["genders"])
        ages = np.array(data.get("ages", []))
        
        n = len(y_true)
        pos = np.sum(y_true)
        
        print("\n" + "="*40)
        print(f"MIMIC-III VALIDATION PROGRESS MONITOR")
        print("="*40)
        print(f"Patients Processed: {n}")
        print(f"Readmissions (Pos): {pos} ({pos/n*100:.2f}%)")
        
        if len(np.unique(y_true)) < 2:
            print("\nWait for more positive cases to calculate ROC/PRC.")
        else:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            
            print(f"\nCURRENT PERFORMANCE (as of last checkpoint):")
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
            print(f"Brier: {brier:.4f}")
            
            # Subgroups
            if len(np.unique(genders)) == 2:
                for label, val in [("Male", 1), ("Female", 0)]:
                    mask = genders == val
                    if np.sum(mask) > 0 and len(np.unique(y_true[mask])) > 1:
                        g_auroc = roc_auc_score(y_true[mask], y_prob[mask])
                        print(f"AUROC ({label}): {g_auroc:.4f}")
        
        print("="*40)
        print("Tip: You can run this script anytime in a second terminal window.")
        print("The main script will NOT be interrupted.\n")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    monitor()
