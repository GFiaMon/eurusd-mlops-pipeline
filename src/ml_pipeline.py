import sys
import os
import importlib.util
from dotenv import load_dotenv

load_dotenv()

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Determine paths
# If running from root: src/ml_pipeline.py -> src/
# If running from src: ./ml_pipeline.py -> ./
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import modules dynamically
try:
    ingest_module = import_module_from_path("ingest_data", os.path.join(SCRIPT_DIR, "01_ingest_data.py"))
    preprocess_module = import_module_from_path("preprocess", os.path.join(SCRIPT_DIR, "02_preprocess.py"))
    train_module = import_module_from_path("train_models", os.path.join(SCRIPT_DIR, "03_train_models.py"))
except Exception as e:
    print(f"CRITICAL: Failed to import modules: {e}")
    sys.exit(1)

def run_pipeline():
    print("="*50)
    print("🚀 STARTING EUR/USD ML PIPELINE")
    print("="*50)
    
    # Step 1: Ingestion
    print("\n[STEP 1] Data Ingestion")
    try:
        ingest_module.ingest_data()
    except Exception as e:
        print(f"❌ Ingestion Failed: {e}")
        return

    # Step 2: Preprocessing
    print("\n[STEP 2] Preprocessing & Feature Engineering")
    try:
        preprocess_module.preprocess_data()
    except Exception as e:
        print(f"❌ Preprocessing Failed: {e}")
        return

    # Step 3: Training
    print("\n[STEP 3] Model Training")
    try:
        train_module.train_models()
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        return
        
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_pipeline()
