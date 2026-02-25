import sys
import os
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from config import MODELS_DIR, FEATURES_CSV
from embedding_utils import get_model_container

print("Loading model...")
model_container = get_model_container()

print("Loading dataset...")
df = pd.read_csv(FEATURES_CSV)

print("Selecting features...")
features = model_container.model_data['features']
df_features = df[[f for f in features if f in df.columns]].fillna(0)
for f in features:
    if f not in df_features.columns:
        df_features[f] = 0

print("Calculating probabilities...")
probas = model_container.predict_proba(df_features)

df['proba'] = probas

# Thresholds from 08_predict.py
THRESHOLD_HIGH_RISK = 0.70
THRESHOLD_MEDIUM_RISK = 0.40

high_risk = df[df['proba'] >= THRESHOLD_HIGH_RISK].head(2)
medium_risk = df[(df['proba'] >= THRESHOLD_MEDIUM_RISK) & (df['proba'] < THRESHOLD_HIGH_RISK)].head(2)
low_risk = df[df['proba'] < THRESHOLD_MEDIUM_RISK].head(2)

def print_patient(row, risk_level):
    print(f"\n--- {risk_level} RISK PATIENT ---")
    print(f"Probability: {row['proba']:.2f}")
    
    # Core features to print
    core_features = ['anchor_age', 'gender', 'los_days', 'prev_admissions', 'admission_type']
    for f in core_features:
        if f in row:
            print(f"{f}: {row[f]}")
            
    # Print some other top features if available
    other_features = ['days_since_last', 'prev_readmit_rate', 'bmi', 'lab_abnormal_count', 'rx_count']
    for f in other_features:
        if f in row:
            print(f"{f}: {row[f]}")

print("\n=== TEST SAMPLES ===")
for _, row in high_risk.iterrows():
    print_patient(row, "HIGH")

for _, row in medium_risk.iterrows():
    print_patient(row, "MEDIUM")
    
for _, row in low_risk.iterrows():
    print_patient(row, "LOW")

