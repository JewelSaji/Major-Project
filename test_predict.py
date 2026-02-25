import sys
import os
import importlib
import logging

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
predict = importlib.import_module("08_predict")

logging.basicConfig(level=logging.INFO)

print("--- Testing Prediction Logic with Note ---")
mock_data = {
    'anchor_age': 60,
    'gender': 1,
    'los_days': 5.0,
    'prev_admissions': 2,
    'admission_type': 1,
    'clinical_note': 'Patient presents with severe shortness of breath, bilateral crackles on auscultation. History of heart failure. Chest X-ray shows pulmonary edema. Administered IV Lasix. Patient stable but requires further diuresis. High risk of readmission.',
    '_debug_payload': False
}

predict.run_inference(mock_data)
