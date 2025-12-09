# EUR/USD Exchange Rate Prediction - Capstone Project

## 📋 Project Overview
Machine learning pipeline for predicting EUR/USD exchange rates with MLOps practices.

## 🎯 Objectives
1. Develop ML models to predict EUR/USD exchange rates
2. Implement MLOops practices (experiment tracking, model registry)
3. Deploy model as API on AWS
4. Create reproducible pipeline

## 📁 Project Structure
```
eurusd-capstone/
├── notebooks/              # Jupyter notebooks (exploration)
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_experiments.ipynb
│   ├── 05_mlflow_tracking.ipynb
│   └── 06_api_testing.ipynb
├── src/                    # Source code
│   ├── data/              # Data collection & processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions
│   ├── visualization/     # Visualization utilities
│   └── utils/             # Utility functions
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data
│   └── models/            # Saved models
├── tests/                  # Unit tests
├── api/                    # API code
├── config/                 # Configuration files
├── mlruns/                 # MLflow experiments
├── figures/                # Generated visualizations
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── ml_pipeline.py          # Main ML pipeline
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd eurusd-capstone

# Create and activate virtual environment
python3.11 -m venv ~/venvs/venv_eurusd
source ~/venvs/venv_eurusd/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name="venv_eurusd" --display-name="EUR/USD Capstone"
```

### 2. Run Data Collection
```python
# In notebook or script
from src.data.collect_data import fetch_eurusd_data
df = fetch_eurusd_data(years=3)
```

### 3. Run ML Pipeline
```bash
python ml_pipeline.py
```

## 📊 Development Workflow
1. **Exploration**: Use notebooks in `notebooks/` directory
2. **Prototyping**: Experiment in notebooks first
3. **Production**: Move working code to `src/` modules
4. **Testing**: Run tests in `tests/` directory
5. **Tracking**: Use MLflow for experiment tracking

## 📅 Project Timeline
- **Day 1**: ML Model Development
- **Day 2**: MLOps Setup
- **Day 3**: API Development
- **Day 4**: AWS Research & Planning
- **Day 5**: AWS Deployment
- **Day 6**: Buffer & Monitoring
- **Day 7**: Documentation & Polish

## 🛠️ Tools & Technologies
- **ML**: scikit-learn, pandas, numpy
- **MLOps**: MLflow
- **API**: FastAPI
- **Deployment**: AWS (SageMaker/EC2)
- **Version Control**: Git, GitHub
- **Visualization**: matplotlib, seaborn

## 📞 Contact
[Your Name]
[Your Email]
[Your GitHub Profile]
