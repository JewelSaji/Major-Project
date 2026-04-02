import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MIMIC3 Dataset")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

def analyze_labels():
    print("Loading cohort_admissions.csv...")
    adm = pd.read_csv(os.path.join(DATA_DIR, "cohort_admissions.csv"), parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"])
    adm = adm.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
    
    # Label construction logic (Matches src/15_mimic3_validation.py)
    adm["next_admittime"] = adm.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
    adm["days_to_next"] = (adm["next_admittime"] - adm["DISCHTIME"]).dt.total_seconds() / 86400
    adm["died_hospital"] = adm["DEATHTIME"].notna()
    
    # Filter for readmitted (readmit_30 == 1)
    positives = adm[(adm["days_to_next"] >= 0) & (adm["days_to_next"] <= 30) & (~adm["died_hospital"])].copy()
    
    print(f"\nTotal Admissions: {len(adm)}")
    print(f"Positive Readmissions (30-day): {len(positives)}")
    print(f"Overall Readmission Rate: {len(positives)/len(adm)*100:.2f}%")
    
    # Stats on days_to_next
    stats = positives["days_to_next"].describe()
    print("\nDays to Next Admission Stats (Positive Cases only):")
    print(stats)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(positives["days_to_next"], bins=30, color="#2c3e50", edgecolor="white", alpha=0.8)
    plt.title("Distribution of Days to Next Admission (MIMIC-III Positive Cases)", fontsize=14)
    plt.xlabel("Days until next admission", fontsize=12)
    plt.ylabel("Count of Patients", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 30)
    
    plot_path = os.path.join(OUT_DIR, "mimic3_readmission_dist.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {plot_path}")

if __name__ == "__main__":
    analyze_labels()
