# Data Architecture & Pipeline Guide

This document explains the "Cloud-First, Local-Mirror" data architecture used in the EUR/USD Capstone project.

## 🏗️ Architecture Overview

The system unifies data access across three environments using a single logic layer:
1.  **AWS Lambda**: Ingests daily data and saves directly to S3 (partitioned files).
2.  **Local Development**: Mirrors S3 data locally for fast iteration and offline work.
3.  **Flask API (EC2)**: Loads historical data from S3 (or local fallback) for predictions.

**Key Principle**: **S3 is the Single Source of Truth**. Local environments mirror S3, they don't diverge from it.

### The `DataManager` Logic
All data access goes through `utils.data_manager.DataManager`. It automatically detects its environment:

| Environment | Detection Logic | Behavior |
| :--- | :--- | :--- |
| **Lambda** | `AWS_LAMBDA_FUNCTION_NAME` env var | Reads/Writes directly to S3 (partitioned `data_{end}_from_{start}.csv`). |
| **EC2** | Instance Metadata | Reads directly from S3. Fallback to local files if S3 fails. |
| **Cloud** | Valid AWS Credentials present | Syncs S3 -> Local Mirror. Reads from Local. Syncs Local -> S3 on save. |
| **Local** | No AWS Credentials | Reads/Writes to local `data/` directory only. |
| **Offline** | `FORCE_OFFLINE=true` | Ignores S3 completely. Uses local cache. |

---

## 🛠️ Workflows

### 1. Data Ingestion (Daily)
**Script**: `src/01_ingest_data.py` (Local) or `aws_lambda/ingest/lambda_function.py` (Cloud)
- **What it does**: Fetches EUR/USD data from Yahoo Finance.
- **Local**: Merges new data with existing `data/raw/eurusd_latest.csv`, saves locally, and syncs to S3.
- **Lambda**: Saves a daily partition `data_{end}_from_{start}.csv` to S3.

### 2. Preprocessing
**Script**: `src/02_preprocess.py`
- **What it does**: Loads raw data, generates features (MA, Lags), scales, and splits.
- **Output**: `data/processed/train.csv`, `data/processed/test.csv`, `data/processed/scaler.pkl`.
- **Sync**: Automatically uploads processed artifacts to S3 if credentials are available.

### 3. Training & Inference
- **Training**: Should load from `data/processed/` using `DataManager(data_type='processed')`.
- **API**: `api/app_cloud.py` initializes `DataManager` to fetch the latest historical data from S3 to generate features for live predictions.

---

## 💻 Developer Guide

### How to use `DataManager`
```python
from utils.data_manager import DataManager

# 1. Get Raw Data
dm = DataManager(data_type='raw')
df = dm.get_latest_data() # Auto-syncs from S3 if needed

# 2. Load Processed Data (Train/Test/Scaler)
dm_proc = DataManager(data_type='processed')
train_df, test_df, scaler = dm_proc.load_processed()

# 3. Save New Data
dm.save_data(df, metadata={'source': 'manual_fix'})
```

### Directory Structure
```
data/
├── raw/
│   ├── eurusd_latest.csv       # The Unified Merged File (Local Mirror)
│   ├── data_2024-12-24...csv   # Daily partitions (naming: data_{end}_from_{start}.csv)
│   └── metadata.json           # Provenance info
└── processed/
    ├── train.csv
    ├── test.csv
    ├── scaler.pkl
    └── metadata.json
```

---

## 🚀 Deployment

The `DataManager` class is the core dependency. Deployment scripts automatically handle it:
- **Flask API**: Copies `utils/data_manager.py` -> `api/utils/` before Docker build.
- **Lambda**: Copies `utils/data_manager.py` -> `aws_lambda/ingest/` before Zipping.

**Note**: Do not modify the copies in `api/` or `aws_lambda/` directly. Always edit `utils/data_manager.py` and run deployment scripts to propagate changes.
