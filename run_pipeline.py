# run_pipeline.py
"""
TRANCE Framework - One-Click Execution Pipeline
Automates the entire workflow from extraction to visualization.
"""

import sys
import os
import subprocess
import time
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_step(name, script_path, args=None):
    """Runs a single pipeline step and tracks timing"""
    start_time = time.time()
    logger.info(f"🚀 STARTING STEP: {name}")
    
    # Use the new numbered filenames
    cmd = [sys.executable, "-m", f"src.{script_path.replace('.py', '')}"]
    if args:
        cmd.extend(args)
        
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        duration = time.time() - start_time
        logger.info(f"✅ COMPLETED STEP: {name} in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FAILED STEP: {name}")
        logger.error(f"Command: {' '.join(cmd)}")
        return False

def main():
    print("\n" + "="*60)
    print("      TRANCE FRAMEWORK - FULL PIPELINE EXECUTION")
    print("="*60 + "\n")

    # ------------------------------------------------------------------ #
    # PIPELINE STEPS (batch-runnable, executed sequentially)              #
    #                                                                     #
    #  01_extract.py           – Extract & engineer features from MIMIC   #
    #  01b_select_features.py  – SHAP-based feature pruning               #
    #  02a_finetune_clinical_t5.py – (optional) LoRA fine-tune ClinicalT5 #
    #  02_embed.py             – Generate ClinicalT5 text embeddings       #
    #  03_train.py             – Train & calibrate ensemble model          #
    #  04_diagnose.py          – Embedding diagnostics / sanity checks     #
    #  05_analyze.py           – SHAP interpretability analysis            #
    #  06_visualize.py         – Journal-quality result visualizations     #
    #  09_compare_models.py    – Cross-paper model benchmark comparison    #
    #                                                                     #
    # NOT INCLUDED (require manual launch after pipeline completes):      #
    #  07_api.py   – FastAPI prediction server (blocking / long-running)  #
    #               Launch with:  python src/07_api.py                    #
    #               or:  uvicorn src.07_api:app --host 0.0.0.0 --port 8000#
    #  08_predict.py – Interactive CLI predictor (requires user input)    #
    #               Launch with:  python src/08_predict.py                #
    # ------------------------------------------------------------------ #

    steps = [
        # Stage 1 – Data preparation
        ("1/9  Feature Extraction",           "01_extract.py"),
        ("2/9  Feature Selection (SHAP)",     "01b_select_features.py"),
        # Stage 2 – Embeddings  (02a is opt-in via env var)
        ("4/9  Clinical T5 Embedding",        "02_embed.py"),
        # Stage 3 – Model training
        ("5/9  Model Training & Calibration", "03_train.py"),
        # Stage 4 – Diagnostics & analysis
        ("6/9  Embedding Diagnostics",        "04_diagnose.py"),
        ("7/9  SHAP Interpretability",        "05_analyze.py"),
        # Stage 5 – Output generation
        ("8/9  Journal Visualizations",       "06_visualize.py"),
        ("9/9  Cross-Paper Model Comparison", "09_compare_models.py"),
    ]

    # Optionally inject ClinicalT5 fine-tuning before embedding step
    if os.environ.get("RUN_CT5_FINETUNE", "0") == "1":
        steps.insert(2, ("3/9  Clinical T5 LoRA Fine-tuning", "02a_finetune_clinical_t5.py"))

    total_start = time.time()
    success_count = 0

    for name, script in steps:
        if run_step(name, script):
            success_count += 1
        else:
            logger.error("🛑 Pipeline halted due to error.")
            break

    total_duration = time.time() - total_start
    print("\n" + "="*60)
    print(f"      PIPELINE SUMMARY")
    print(f"      Steps Completed: {success_count}/{len(steps)}")
    print(f"      Total Duration:  {total_duration/60:.2f} minutes")
    print("="*60 + "\n")

    if success_count == len(steps):
        print("🎉 SUCCESS: The entire TRANCE pipeline has been executed successfully.")
        print()
        print("  Next steps (launch manually):")
        print("  ┌──────────────────────────────────────────────────────────┐")
        print("  │  API Server   : python src/07_api.py                    │")
        print("  │  CLI Predictor: python src/08_predict.py                │")
        print("  └──────────────────────────────────────────────────────────┘")
    else:
        print("⚠️  WARNING: Pipeline finished with errors.")


if __name__ == "__main__":
    main()

