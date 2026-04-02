import os
import sys
import gc
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK
from embedding_utils import get_embedding, get_model_container

# MIMIC-3 CareVue cohort dir
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MIMIC3 Dataset")

# Expected Feature Map (as defined in MIMIC-IV extract)
KEY_LAB_ITEMS = {
    50912: "creatinine", 50902: "chloride",  50882: "bicarb",
    50931: "glucose",    50971: "potassium", 50983: "sodium",
    51006: "bun",        51221: "hematocrit",51222: "hemoglobin",
    51265: "platelets",  51301: "wbc",       50813: "lactate",
    50820: "ph",         50821: "pao2",      50818: "paco2",
}

VITAL_ITEMS = {
    211: "hr", 
    51: "sbp", 455: "sbp", 
    8368: "sbp", # Arterial BP [Systolic]
    676: "temp_c", 678: "temp_f",
    618: "rr", 615: "rr",
}

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            prob_mean = np.mean(y_prob[mask])
            acc_mean = np.mean(y_true[mask])
            ece += np.abs(prob_mean - acc_mean) * np.sum(mask)
    return ece / len(y_prob)


def _comorbidities(dx):
    cm_map = {
        "chf": r"^(428|I50)", "arrhythmia": r"^(427|I4[7-9])", "diabetes": r"^(250|E1[0-4])",
        "hypertension": r"^(40[1-5]|I1[0-6])", "renal_fail": r"^(585|586|N1[89])",
        "copd": r"^(49[0-6]|J4[1-6])", "liver": r"^(57[0-3]|K7[0-4])",
        "cancer": r"^(1[4-9][0-9]|C[0-9])", "depression": r"^(296|311|F3[2-4])",
        "psychosis": r"^(295|297|298|F2[0-9])", "obesity": r"^(278|E66)",
        "sepsis": r"^(99591|99592|A41|R65)", "pneumonia": r"^(48[0-6]|J1[2-8])",
        "stroke": r"^(43[0-8]|I6[0-9])", "mi": r"^(410|41[0-2]|I2[1-2])",
        "dementia": r"^(290|331|F0[0-3])",
    }
    dx["icd_s"] = dx["ICD9_CODE"].astype(str)
    cm_dict = {h: {} for h in dx["HADM_ID"].unique()}
    
    for name, pat in cm_map.items():
        has = dx[dx["icd_s"].str.contains(pat, na=False, regex=True)]["HADM_ID"].unique()
        for h in has:
            cm_dict[h][f"cm_{name}"] = 1.0
            
    return pd.DataFrame.from_dict(cm_dict, orient='index').fillna(0).reset_index().rename(columns={"index": "hadm_id"})


def run_pipeline():
    logger.info("Starting MIMIC-III Temporal Validation Pipeline...")
    model_container = get_model_container()
    if not model_container.model_data:
        raise ValueError("Model not found.")
        
    model_features = model_container.model_data["features"]
    
    # 1. Load Adm & Build target
    logger.info("Loading cohort_admissions.csv...")
    adm = pd.read_csv(os.path.join(DATA_DIR, "cohort_admissions.csv"), parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"])
    adm = adm.sort_values(["SUBJECT_ID", "ADMITTIME"]).reset_index(drop=True)
    
    # Label construction
    adm["next_admittime"] = adm.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
    adm["days_to_next"] = (adm["next_admittime"] - adm["DISCHTIME"]).dt.total_seconds() / 86400
    adm["died_hospital"] = adm["DEATHTIME"].notna()
    
    # Simple proxy: next admission within 30 days and patient didn't die
    adm["readmit_30"] = ((adm["days_to_next"] >= 0) & (adm["days_to_next"] <= 30) & (~adm["died_hospital"])).astype(int)
    
    # Calculate Age
    adm["DOB"] = pd.to_datetime(adm["DOB"], errors="coerce")
    # Calculate Age robustly to avoid OverflowError with MIMIC-3 de-identification offsets
    adm["ADMIT_YEAR"] = adm["ADMITTIME"].dt.year
    adm["DOB_YEAR"] = adm["DOB"].dt.year
    adm["anchor_age"] = adm["ADMIT_YEAR"] - adm["DOB_YEAR"]
    
    # MIMIC-3 specific: >89 age is masked to 300+, reset to 90
    adm.loc[adm["anchor_age"] > 100, "anchor_age"] = 90.0
    
    adm["gender"] = adm["GENDER"].map({"M": 1, "F": 0}).fillna(0)
    adm["los_hours"] = (adm["DISCHTIME"] - adm["ADMITTIME"]).dt.total_seconds() / 3600
    adm["los_days"] = adm["los_hours"] / 24
    
    # Save a mapped DF
    df = pd.DataFrame()
    df["hadm_id"] = adm["HADM_ID"]
    df["subject_id"] = adm["SUBJECT_ID"]
    df["readmit_30"] = adm["readmit_30"]
    df["anchor_age"] = adm["anchor_age"].clip(18, 120)
    df["gender"] = adm["gender"]
    df["los_days"] = adm["los_days"].clip(0.1, 120)
    df["los_hours"] = adm["los_hours"].clip(1, 2880)
    
    # Calculate previous admissions strictly using chronological order
    df["prev_admissions"] = adm.groupby("SUBJECT_ID").cumcount()
    
    # 2. Vitals
    logger.info("Loading vitals_carevue.csv...")
    vitals = pd.DataFrame()
    if os.path.exists(os.path.join(DATA_DIR, "vitals_carevue.csv")):
        vinfo = pd.read_csv(os.path.join(DATA_DIR, "vitals_carevue.csv"))
        vinfo["vname"] = vinfo["ITEMID"].map(VITAL_ITEMS)
        # Convert C to F
        vinfo.loc[vinfo["vname"] == "temp_c", "VALUENUM"] = (vinfo.loc[vinfo["vname"] == "temp_c", "VALUENUM"] * 9/5) + 32
        vinfo.loc[vinfo["vname"] == "temp_c", "vname"] = "temp_f"
        
        vdf = []
        for h, grp in vinfo.groupby("HADM_ID"):
            row = {"hadm_id": h}
            for vname, vgrp in grp.groupby("vname"):
                vals = vgrp["VALUENUM"].values
                row[f"v_{vname}_mean"] = np.mean(vals)
                row[f"v_{vname}_min"] = np.min(vals)
                row[f"v_{vname}_max"] = np.max(vals)
                row[f"v_{vname}_std"] = np.std(vals) if len(vals) > 1 else 0.0
            vdf.append(row)
        vitals = pd.DataFrame(vdf)
        df = df.merge(vitals, on="hadm_id", how="left")
        
    # 3. Labs
    logger.info("Loading labs_carevue.csv...")
    labs = pd.DataFrame()
    if os.path.exists(os.path.join(DATA_DIR, "labs_carevue.csv")):
        linfo = pd.read_csv(os.path.join(DATA_DIR, "labs_carevue.csv"))
        linfo["lname"] = linfo["ITEMID"].map(KEY_LAB_ITEMS)
        ldf = []
        for h, grp in linfo.groupby("HADM_ID"):
            row = {"hadm_id": h}
            for lname, lgrp in grp.groupby("lname"):
                vals = lgrp["VALUENUM"].values
                row[f"lab_{lname}_mean"] = np.mean(vals)
                row[f"lab_{lname}_min"] = np.min(vals)
                row[f"lab_{lname}_max"] = np.max(vals)
                row[f"lab_{lname}_last"] = vals[-1] if len(vals) > 0 else 0.0
            ldf.append(row)
        labs = pd.DataFrame(ldf)
        df = df.merge(labs, on="hadm_id", how="left")
        
    # 4. Diagnoses
    logger.info("Loading dx_proc_carevue.csv...")
    cm_df = pd.DataFrame()
    if os.path.exists(os.path.join(DATA_DIR, "dx_proc_carevue.csv")):
        dx_info = pd.read_csv(os.path.join(DATA_DIR, "dx_proc_carevue.csv"))
        cm_df = _comorbidities(dx_info)
        df = df.merge(cm_df, on="hadm_id", how="left")
        
    # 5. Notes Embedding (this is the key difference from eicu)
    logger.info("Loading notes_carevue.csv...")
    notes_dict = {}
    if os.path.exists(os.path.join(DATA_DIR, "notes_carevue.csv")):
        notes_info = pd.read_csv(os.path.join(DATA_DIR, "notes_carevue.csv"))
        notes_dict = dict(zip(notes_info["HADM_ID"], notes_info["TEXT"]))
        
    # Extract subset to save time (we want around 10k rows for robust testing, but 500 is good for quick run)
    logger.info(f"Loaded total {len(df)} admissions from CareVue.")
    
    # Checkpoint setup
    CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics", "mimic3_validation_checkpoint.json")
    
    def save_checkpoint(indices, y_t, y_p, gens, ags):
        checkpoint = {
            "processed_indices": [int(i) for i in indices],
            "y_true": [int(i) for i in y_t],
            "y_prob": [float(i) for i in y_p],
            "genders": [int(i) for i in gens],
            "ages": [float(i) for i in ags]
        }
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        logger.info(f"Checkpoint saved: {len(indices)} patients processed at {datetime.now()}")

    checkpoint_data = None
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Found existing checkpoint at {CHECKPOINT_PATH}. Loading...")
        with open(CHECKPOINT_PATH, "r") as f:
            checkpoint_data = json.load(f)

    # Re-apply DEFAULTS
    for feat in model_features:
        if feat not in df.columns:
            if feat in DEFAULTS:
                df[feat] = DEFAULTS[feat]
            elif feat.startswith("ct5_"):
                df[feat] = 0.0 # Will be populated
            else:
                df[feat] = 0.0
                
    df.fillna(0, inplace=True)
    
    # 6. Generate Embeddings dynamically per row
    logger.info(f"Running inference on FULL cohort ({len(df)} patients). This will take time...")
    
    y_true = []
    y_prob = []
    genders = []
    ages = []
    processed_indices = []
    
    if checkpoint_data:
        y_true = checkpoint_data["y_true"]
        y_prob = checkpoint_data["y_prob"]
        genders = checkpoint_data["genders"]
        ages = checkpoint_data.get("ages", [])
        processed_indices = checkpoint_data["processed_indices"]
        logger.info(f"Resuming from checkpoint: {len(processed_indices)} patients already processed.")

    # Loop over ALL rows
    from tqdm import tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        if idx in processed_indices:
            continue
            
        hid = row["hadm_id"]
        note = str(notes_dict.get(hid, ""))
        
        feat_dict = row.to_dict()
        if len(note.strip()) > 50:
            # this produces 128 embedding floats + updates lengths
            emb = get_embedding(text=note, features=feat_dict)
            for i, val in enumerate(emb):
                feat_dict[f"ct5_{i}"] = float(val)
            feat_dict["ct5_has_note"] = 1.0
            feat_dict["ct5_note_len_chars"] = float(np.log1p(len(note)))
            feat_dict["ct5_note_len_tokens"] = float(np.log1p(len(note.split())))
        else:
            feat_dict["ct5_has_note"] = 0.0
            feat_dict["ct5_note_len_chars"] = 0.0
            feat_dict["ct5_note_len_tokens"] = 0.0
            
        # Select exact model features
        x_row = {f: feat_dict.get(f, 0.0) for f in model_features}
        X = pd.DataFrame([x_row], columns=model_features)
        
        try:
            prob = float(model_container.predict_proba(X)[0])
            y_true.append(int(feat_dict["readmit_30"]))
            y_prob.append(float(prob))
            genders.append(int(feat_dict["gender"]))
            ages.append(float(feat_dict["anchor_age"]))
            processed_indices.append(idx)
            
            # Save checkpoint every 2,000 patients
            if len(processed_indices) % 2000 == 0:
                save_checkpoint(processed_indices, y_true, y_prob, genders, ages)
                
        except Exception as e:
            logger.error(f"Prediction failed for row {idx}: {e}")
            
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    genders = np.array(genders)
    ages = np.array(ages)
    
    # Evaluation
    logger.info("Computing final metrics...")
    def bootstrap_metric(metric_fn, y_t, y_p, n_bootstraps=1000):
        scores = []
        n = len(y_t)
        if n == 0: return 0.0, 0.0, 0.0
        for i in range(n_bootstraps):
            np.random.seed(i+42)
            idx = np.random.randint(0, n, n)
            if len(np.unique(y_t[idx])) < 2:
                continue
            scores.append(metric_fn(y_t[idx], y_p[idx]))
        
        if not scores:
            return 0.0, 0.0, 0.0
        return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

    res = {
        "Cohort": "MIMIC-III CareVue (Full Temporal Validation)",
        "N_Samples": len(y_true),
        "Pos_Rate": float(np.mean(y_true)),
        "Execution_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    auroc, au_lb, au_ub = bootstrap_metric(roc_auc_score, y_true, y_prob)
    auprc, ap_lb, ap_ub = bootstrap_metric(average_precision_score, y_true, y_prob)
    brier, br_lb, br_ub = bootstrap_metric(brier_score_loss, y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    
    res["AUROC"] = {"mean": float(auroc), "ci_lower": float(au_lb), "ci_upper": float(au_ub)}
    res["AUPRC"] = {"mean": float(auprc), "ci_lower": float(ap_lb), "ci_upper": float(ap_ub)}
    res["Brier"] = {"mean": float(brier), "ci_lower": float(br_lb), "ci_upper": float(br_ub)}
    res["ECE"]   = {"mean": float(ece)}
    
    # Subgroups
    # Gender
    if len(np.unique(genders)) == 2:
        idx_m = genders == 1
        idx_f = genders == 0
        if np.sum(idx_m) > 0 and np.sum(idx_f) > 0:
            if len(np.unique(y_true[idx_m])) > 1:
                res["AUROC_Male"] = float(roc_auc_score(y_true[idx_m], y_prob[idx_m]))
            if len(np.unique(y_true[idx_f])) > 1:
                res["AUROC_Female"] = float(roc_auc_score(y_true[idx_f], y_prob[idx_f]))
    
    # Age Groups
    age_bins = [0, 30, 50, 70, 90, 120]
    age_labels = ["18-30", "30-50", "50-70", "70-90", "90+"]
    for i in range(len(age_bins)-1):
        mask = (ages >= age_bins[i]) & (ages < age_bins[i+1])
        if np.sum(mask) > 10 and len(np.unique(y_true[mask])) > 1:
            res[f"AUROC_Age_{age_labels[i]}"] = float(roc_auc_score(y_true[mask], y_prob[mask]))

    # Save final results
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mimic3_temporal_validation.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    
    # Remove checkpoint after successful completion
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Final completion: removed checkpoint file.")
        
    logger.info("External Validation Complete. Results:")
    print(json.dumps(res, indent=2))
    
if __name__ == "__main__":
    run_pipeline()
