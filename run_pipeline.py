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
    
    steps = [
        ("Feature Extraction", "01_extract.py"),
        ("Feature Selection (SHAP)", "01b_select_features.py"),
    ]

    if os.environ.get("RUN_CT5_FINETUNE", "0") == "1":
        steps.append(("Clinical T5 LoRA Fine-tuning", "02a_finetune_clinical_t5.py"))

    steps.extend([
        ("Clinical T5 Embedding", "02_embed.py"),
        ("Model Training & Calibration", "03_train.py"),
        ("Embedding Diagnostics", "04_diagnose.py"),
        ("SHAP Interpretability", "05_analyze.py"),
        ("Journal Visualizations", "06_visualize.py"),
        ("Cross-Paper Model Comparison", "09_compare_models.py"),
    ])
    
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
    else:
        print("⚠️ WARNING: Pipeline finished with errors.")

if __name__ == "__main__":
    main()
