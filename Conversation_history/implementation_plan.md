# Implementation Plan - EUR/USD Model Analysis & Improvement

## Current Situation

After analyzing all notebooks, I've found:

### Existing Models:
1. **ARIMA** (`00_data_tests.ipynb`) - Experimental implementation, AutoARIMA found best order (0,0,0)
2. **Linear Regression** (`05_2_model_comparison.ipynb`) - **Currently best performing model**
3. **LSTM Models** (both `04_1_model_experiments.ipynb` and `04_2_model_experiments_lstm_simple.ipynb`):
   - Complex 4-layer LSTM (`04_1`): R² = negative
   - Simplified 2-layer LSTM (`04_2`): R² = **-3.4407** (extremely poor)

### Feature Engineering:
- `03_1_feature_engineering.ipynb`: Complex (40+ features) - used by Linear Regression
- `03_2_feature_engineering_simple.ipynb`: Simple (only closing prices) - used by LSTM

### The Problem:
Your LSTM model has **negative R² values**, meaning it performs **worse than simply predicting the mean**. This is a serious issue that needs investigation.

## User Review Required

> [!WARNING]
> **Your LSTM model is not learning properly** - R² of -3.44 indicates the model is making very poor predictions. Simply reducing layers won't fix this fundamental issue.

> [!IMPORTANT]
> **Linear Regression is currently your best model** - Before investing time in LSTM improvements, we should understand WHY it's failing and whether LSTM is appropriate for this problem.

> [!CAUTION]
> **Key Questions to Address:**
> 1. Do you want to fix the LSTM or focus on improving Linear Regression?
> 2. The negative R² suggests possible data leakage, scaling issues, or that LSTM isn't suitable for this data
> 3. Your instructor's advice to "increase layers" may not apply here - the issue is more fundamental

## Proposed Diagnostic & Improvement Plan

### Phase 1: Understand Why LSTM is Failing (RECOMMENDED FIRST STEP)

#### Diagnostic Checks:
1. **Data Leakage Check**
   - Verify train/test split is chronological
   - Ensure scaler is fit ONLY on training data
   - Check for any future information in features

2. **Scaling Issues**
   - Verify MinMaxScaler is applied correctly
   - Check if predictions are being inverse-transformed properly
   - Ensure no NaN or infinite values in scaled data

3. **Model Architecture Issues**
   - Current 2-layer LSTM may be too simple OR too complex
   - Check if model is actually learning (loss decreasing?)
   - Verify input shape matches data shape

4. **Data Suitability**
   - EUR/USD may be too noisy for LSTM with only 779 samples
   - Consider if Linear Regression is simply more appropriate

### Phase 2: Choose Your Path Forward

#### Option A: Fix LSTM (If you must use it for your project)

**[MODIFY]** [04_2_model_experiments_lstm_simple.ipynb](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/notebooks/04_2_model_experiments_lstm_simple.ipynb)
- Add diagnostic cells to check data quality
- Verify scaling is correct
- Try different architectures:
  - Single LSTM layer (simpler)
  - Different activation functions
  - Different optimizers (try SGD instead of Adam)
- Add validation plots to see if model is learning
- Compare predictions vs actual values visually

**[MODIFY]** [03_2_feature_engineering_simple.ipynb](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/notebooks/03_2_feature_engineering_simple.ipynb)
- Verify train/test split is correct
- Add data quality checks
- Ensure no data leakage

#### Option B: Improve Linear Regression (Currently your best model)

**[MODIFY]** [05_2_model_comparison.ipynb](file:///Users/guillermo/Documents/Ironhack/M8_Capstone/2-Capstone-2_MLOps/1-Development/eurusd-capstone/notebooks/05_2_model_comparison.ipynb)
- Feature selection to improve Linear Regression
- Try Ridge/Lasso regression for regularization
- Ensemble methods combining Linear Regression with other models
- Document why Linear Regression works better

#### Option C: Explore ARIMA Further

**[NEW]** Create `04_3_arima_model.ipynb`
- Move ARIMA code from `00_data_tests.ipynb` to proper notebook
- Implement proper train/test evaluation
- Compare with Linear Regression

## Verification Plan

### Automated Tests
- I will run the notebooks using `run_command` (converting them to scripts or running via `ipython` if needed, or just manually running cells via a browser checking tool if easier, but `ipython` execution is preferred for reliability).
- I will check for the existence of output files:
    - `data/processed/lstm_sequences.pkl`
    - `models/lstm_best_model.keras`

### Manual Verification
- Inspect the loss curves in the generated notebook output to ensure the model is learning (loss decreasing).
- Check the "Next Day Prediction" output to ensure it's a reasonable EUR/USD price (e.g., around 1.05-1.20).
