"""
test_predict.py — TRANCE Framework prediction tests.

Run from the project root:
    venv/bin/python test_predict.py

Tests 10 patient profiles spanning low → medium → high clinical risk.
Uses mean-based baseline (no nearest-template anchoring) so raw feature
values are fully reflected in the output probability.

Results are saved to results/test_predict_<timestamp>.csv  (one row per patient)
                     results/test_predict_<timestamp>.json (includes run summary)
"""

import sys
import os
import json
import importlib.util
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# ── Setup ─────────────────────────────────────────────────────────
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _load_predict_module():
    """Load src/08_predict.py cleanly, clearing any cached modules first."""
    for mod in list(sys.modules.keys()):
        if any(k in mod for k in ("config", "predict", "embedding_utils")):
            del sys.modules[mod]

    path = os.path.join(project_dir, "src", "08_predict.py")
    spec = importlib.util.spec_from_file_location("predict_module", path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["predict_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_direct(predict_mod, desc: str, overrides: dict, note: str | None = None) -> dict:
    """
    Run inference with a mean-based feature baseline (no template anchoring).
    Returns a result dict with desc, probability, and risk label.
    """
    from src.config import DEFAULTS, THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK

    mc = predict_mod.model_container
    if not mc.model_data:
        print("  [ERROR] Model not loaded — run src/03_train.py first.")
        return {"desc": desc, "probability": -1.0, "risk": "ERROR", "has_note": bool(note)}

    model_features = mc.model_data["features"]
    feature_means  = mc.model_data.get("feature_means", {})
    ref_means      = predict_mod._load_reference_feature_means(model_features)

    # Build baseline: prefer raw dataset means over model means
    full = {k: float(v) for k, v in (ref_means or feature_means or DEFAULTS).items()}
    full.update({k: float(v) for k, v in DEFAULTS.items()})
    full.update({k: float(v) for k, v in overrides.items()})
    predict_mod._recompute_engineered_features(full)

    if note:
        emb = predict_mod.get_embedding(text=note, features=full)
        for i, val in enumerate(emb):
            full[f"ct5_{i}"] = float(val)
        full["ct5_has_note"]        = 1.0
        full["ct5_note_len_chars"]  = float(np.log1p(len(note)))
        full["ct5_note_len_tokens"] = float(np.log1p(len(note.split())))
    else:
        full["ct5_has_note"]        = 0.0
        full["ct5_note_len_chars"]  = 0.0
        full["ct5_note_len_tokens"] = 0.0

    row   = {f: full.get(f, 0.0) for f in model_features}
    X     = pd.DataFrame([row])[model_features]
    proba = float(mc.predict_proba(X)[0])

    if proba >= THRESHOLD_HIGH_RISK:
        risk, sym = "HIGH", "!!!"
    elif proba >= THRESHOLD_MEDIUM_RISK:
        risk, sym = "MEDIUM", "?? "
    else:
        risk, sym = "LOW", "OK "

    bar_len = int(proba * 40)
    bar     = "█" * bar_len + "░" * (40 - bar_len)
    print(f"  {sym} [{bar}] {proba:5.1%}  {risk:<6}  {desc}")

    return {
        "desc":        desc,
        "probability": round(proba, 4),
        "risk":        risk,
        "has_note":    bool(note),
        **{k: overrides.get(k) for k in (
            "anchor_age", "gender", "los_days", "prev_admissions",
            "admission_type", "had_icu", "dx_count", "proc_count",
            "rx_count", "lab_abnormal_count", "lab_abnormal_rate",
            "days_since_last", "prev_readmit_rate",
        ) if k in overrides},
    }


# ── 10 Patient Profiles ───────────────────────────────────────────
PROFILES = [
    # ── LOW risk ─────────────────────────────────────────────────
    {
        "desc": "P01 │ 22yr, 1-day appendectomy, healthy",
        "ov": dict(
            anchor_age=22, gender=0, los_days=1.0,
            prev_admissions=0, admission_type=3,
            dx_count=1, proc_count=1, rx_count=5,
            lab_abnormal_count=0, lab_abnormal_rate=0.0,
            had_icu=0, days_since_last=3650, prev_readmit_rate=0.0,
        ),
        "note": "22-year-old female admitted for elective appendectomy. No complications. Good general health.",
    },
    {
        "desc": "P02 │ 34yr, 2-day childbirth complication",
        "ov": dict(
            anchor_age=34, gender=0, los_days=2.0,
            prev_admissions=1, admission_type=2,
            dx_count=2, proc_count=2, rx_count=10,
            lab_abnormal_count=2, lab_abnormal_rate=0.1,
            had_icu=0, days_since_last=730, prev_readmit_rate=0.0,
        ),
        "note": "34-year-old admitted for minor postpartum complication. Resolved with antibiotics. Discharged well.",
    },
    {
        "desc": "P03 │ 45yr, 3-day elective knee replacement, stable DM",
        "ov": dict(
            anchor_age=45, gender=1, los_days=3.0,
            prev_admissions=1, admission_type=3,
            dx_count=3, proc_count=2, rx_count=20,
            lab_abnormal_count=5, lab_abnormal_rate=0.15,
            had_icu=0, days_since_last=500, prev_readmit_rate=0.1,
        ),
        "note": "45-year-old male undergoing elective knee replacement. Well-controlled diabetes. Uneventful recovery.",
    },
    {
        "desc": "P04 │ 58yr, 5-day pneumonia, 2 prior admits",
        "ov": dict(
            anchor_age=58, gender=1, los_days=5.0,
            prev_admissions=2, admission_type=1,
            dx_count=4, proc_count=3, rx_count=40,
            lab_abnormal_count=20, lab_abnormal_rate=0.35,
            had_icu=0, days_since_last=180, prev_readmit_rate=0.2,
        ),
        "note": "58-year-old with community-acquired pneumonia. Mild hypoxia on admission. Improving with antibiotics.",
    },
    # ── MEDIUM risk ──────────────────────────────────────────────
    {
        "desc": "P05 │ 62yr, 7-day COPD exacerbation, smoker",
        "ov": dict(
            anchor_age=62, gender=1, los_days=7.0,
            prev_admissions=4, admission_type=1,
            dx_count=6, proc_count=4, rx_count=60,
            lab_abnormal_count=30, lab_abnormal_rate=0.40,
            had_icu=1, icu_los_sum=2.0,
            days_since_last=90, prev_readmit_rate=0.35,
        ),
        "note": "62-year-old male smoker with severe COPD exacerbation. On nebulizers and steroids. Brief ICU stay for NIV.",
    },
    {
        "desc": "P06 │ 70yr, 9-day CHF decompensation, obese",
        "ov": dict(
            anchor_age=70, gender=0, los_days=9.0,
            prev_admissions=5, admission_type=1,
            dx_count=8, proc_count=6, rx_count=100,
            lab_abnormal_count=45, lab_abnormal_rate=0.50,
            had_icu=1, icu_los_sum=3.0,
            days_since_last=60, prev_readmit_rate=0.45,
        ),
        "note": "70-year-old obese female with CHF and afib. Acute decompensation with fluid overload. On IV diuretics.",
    },
    {
        "desc": "P07 │ 65yr, 12-day post-op sepsis, DM + CKD3",
        "ov": dict(
            anchor_age=65, gender=1, los_days=12.0,
            prev_admissions=4, admission_type=1,
            dx_count=9, proc_count=10, rx_count=120,
            lab_abnormal_count=60, lab_abnormal_rate=0.55,
            had_icu=1, icu_los_sum=5.0,
            days_since_last=45, prev_readmit_rate=0.50,
        ),
        "note": "65-year-old with post-operative sepsis, diabetes, and stage 3 CKD. Managed with broad-spectrum antibiotics.",
    },
    # ── HIGH risk ────────────────────────────────────────────────
    {
        "desc": "P08 │ 76yr, 18-day advanced CHF (EF 15%), 8 admits",
        "ov": dict(
            anchor_age=76, gender=0, los_days=18.0,
            prev_admissions=8, admission_type=1,
            dx_count=15, proc_count=12, rx_count=200,
            lab_abnormal_count=90, lab_abnormal_rate=0.68,
            had_icu=2, icu_los_sum=8.0,
            days_since_last=7, prev_readmit_rate=0.9, prev_los_mean=14.0,
        ),
        "note": (
            "76-year-old female with advanced CHF (EF 15%), afib, severe pulmonary hypertension. "
            "Admitted via ED with acute decompensation. 3rd readmission this month. "
            "Multiple pressors. High diuretic resistance noted."
        ),
    },
    {
        "desc": "P09 │ 68yr, 20-day ESRD + sepsis, 12 admits, 85% abnormal labs",
        "ov": dict(
            anchor_age=68, gender=1, los_days=20.0,
            prev_admissions=12, admission_type=1,
            dx_count=18, proc_count=25, rx_count=300,
            lab_abnormal_count=200, lab_abnormal_rate=0.85,
            had_icu=2, icu_los_sum=12.0,
            days_since_last=10, prev_readmit_rate=0.95, prev_los_mean=15.0,
        ),
        "note": (
            "68-year-old male with ESRD on dialysis, diabetic foot infection progressing to sepsis. "
            "Blood cultures positive. ICU admission. Frequent hospitalizations last 2 years."
        ),
    },
    {
        "desc": "P10 │ 91yr, 25-day multi-organ failure, palliative care",
        "ov": dict(
            anchor_age=91, gender=1, los_days=25.0,
            prev_admissions=10, admission_type=1,
            dx_count=20, proc_count=20, rx_count=180,
            lab_abnormal_count=150, lab_abnormal_rate=0.80,
            had_icu=3, icu_los_sum=15.0,
            days_since_last=15, prev_readmit_rate=0.85, prev_los_mean=18.0,
        ),
        "note": (
            "91-year-old male with multi-organ failure, sepsis, COPD, CHF, DM, and stage 4 CKD. "
            "Repeated intubation. Palliative care involved. Extremely high readmission risk."
        ),
    },
]


def save_results(results: list[dict], out_dir: str) -> None:
    """Save per-patient results to a timestamped CSV and JSON in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(out_dir, f"test_predict_{ts}")

    # CSV — one row per patient
    df = pd.DataFrame(results)
    csv_path = stem + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV  saved → {csv_path}")

    # JSON — per-patient list + summary totals
    from src.config import THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK
    summary = {
        "run_timestamp": ts,
        "threshold_high":   THRESHOLD_HIGH_RISK,
        "threshold_medium": THRESHOLD_MEDIUM_RISK,
        "n_high":   sum(1 for r in results if r["risk"] == "HIGH"),
        "n_medium": sum(1 for r in results if r["risk"] == "MEDIUM"),
        "n_low":    sum(1 for r in results if r["risk"] == "LOW"),
        "patients": results,
    }
    json_path = stem + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  JSON saved → {json_path}")


def main():
    predict_mod = _load_predict_module()

    from src.config import THRESHOLD_HIGH_RISK, THRESHOLD_MEDIUM_RISK

    print("\n" + "=" * 80)
    print("  TRANCE — 10 PATIENT PROFILE PREDICTION TEST")
    # print(f"  Thresholds: HIGH ≥ {THRESHOLD_HIGH_RISK:.0%}  |  MEDIUM ≥ {THRESHOLD_MEDIUM_RISK:.0%}")
    print("=" * 80)

    results = []
    for p in PROFILES:
        result = run_direct(predict_mod, p["desc"], p["ov"], p.get("note"))
        results.append(result)

    high   = sum(1 for r in results if r["risk"] == "HIGH")
    medium = sum(1 for r in results if r["risk"] == "MEDIUM")
    low    = sum(1 for r in results if r["risk"] == "LOW")

    print("=" * 80)
    print(f"  Summary: {high} HIGH  |  {medium} MEDIUM  |  {low} LOW  (out of {len(results)} patients)")
    print("=" * 80)

    # Save results
    out_dir = os.path.join(project_dir, "results")
    save_results(results, out_dir)
    print()


if __name__ == "__main__":
    main()
