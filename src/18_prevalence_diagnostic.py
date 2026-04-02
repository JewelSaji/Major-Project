import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MIMIC3 Dataset")

def diagnose_prevalence():
    print("Loading cohort_admissions.csv...")
    adm = pd.read_csv(os.path.join(DATA_DIR, "cohort_admissions.csv"), parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"])
    adm = adm.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
    # FILTER FOR ADULTS (MIMIC-IV model is an adult model)
    adm = adm[adm["ADMISSION_TYPE"] != "NEWBORN"].reset_index(drop=True)
    print(f"Adult Admissions: {len(adm)}")

    # Label construction logic (Matching src/01_extract.py exactly)
    adm["next_admittime"] = adm.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
    adm["next_adm_type"]  = adm.groupby("SUBJECT_ID")["ADMISSION_TYPE"].shift(-1)
    adm["days_to_next"]   = (adm["next_admittime"] - adm["DISCHTIME"]).dt.total_seconds() / 86400
    adm["died_hospital"]  = adm["DEATHTIME"].notna()
    
    # MIMIC-IV Logic: Exclude planned next admissions
    planned = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}
    adm["next_planned"] = adm["next_adm_type"].isin(planned)
    
    adm["readmit_30"] = (
        (adm["days_to_next"] >= 0) & (adm["days_to_next"] <= 30) &
        (~adm["died_hospital"]) & (~adm["next_planned"])
    ).fillna(False).astype(int)
    
    print(f"Final Adult Readmission Prevalence: {adm['readmit_30'].mean()*100:.2f}%")
    
    # Breakdown of index admissions vs readmissions
    print("\nIndex Admission Type Breakdown:")
    print(adm.groupby("ADMISSION_TYPE")["readmit_30"].agg(["count", "sum", "mean"]))

    # Look at "True" Unplanned Rate (Index = Emergency/Urgent)
    unplanned_idx = adm[adm["ADMISSION_TYPE"].isin(["EMERGENCY", "URGENT"])]
    print(f"\nUnplanned-Index Readmission Prevalence: {unplanned_idx['readmit_30'].mean()*100:.2f}%")
    
if __name__ == "__main__":
    diagnose_prevalence()
