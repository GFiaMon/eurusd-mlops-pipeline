# 🧠 The Brain of the Pipeline: MLflow Architecture

**Goal:** Automate the battle between our models.

---

## 1. The Challenge 🎯

We have **3 competing architectures**:
*   📉 **Linear Regression** (Simple, Interpretable)
*   📊 **ARIMA** (Statistical, Time-Series focused)
*   🧠 **LSTM** (Deep Learning, Complex)

**The Problem:**
Every day, a different model might be the winner.
*   *How do we deploy the winner automatically?*
*   *How do we ensure we don't deploy a bad model?*

---

## 2. The Solution: "Championship" Logic 🏆

We don't guess. We measure.

*   **Standardized Metrics:** Every model logs `rmse`, `mae`, and `directional_accuracy`.
*   **The Referee:** `src/04_evaluate_select.py`

### 🔴 LIVE DEMO: Experiments (The "Storage") 🗄️
*   [ ] Open **MLflow UI** -> **Experiments** -> `EURUSD_Experiments`
*   [ ] Explain: **"Experiments store the Runs (Artifacts/Metrics)."**
*   [ ] **Sort by RMSE** (Ascending).
*   [ ] Show **Tags**: `model_type`.

> **Crucial Concept:** Deleting a run here *does not* fix the Registry. They are separate decoupled systems.

---

## 3. The Registry: The "VIP List" 📋

The Registry is for **Registered Models** only (pushed from Experiments).
Deleting a model from the registry *does not* delete the run. You often need to clean both if you made a mistake.

**We use Aliases (No Stages):**

| Alias | Meaning |
| :--- | :--- |
| 🥇 `@champion` | The absolute best model (Lowest RMSE). **Deployed.** |
| 🥈 `@challenger` | The runner-up. Good for A/B testing. |
| 🥉 `@candidate` | The third place. |

### 🔴 LIVE DEMO: Registry
*   [ ] Open **MLflow UI** -> **Models**
*   [ ] Click `EURUSD_LSTM` (or any model).
*   [ ] Point to the **Aliases** column.
*   *Note how `@champion` is attached to a specific version.*

---

## 4. The "Smart" Flask API 🧠

Our API is dumb but obedient. It doesn't know "LSTM". It only knows **Champion**.

**The Logic (`api/app_cloud.py`):**
1.  Read `models/best_model_info.json`
2.  Ask MLflow: *"Give me the `@champion`"*
3.  Load `feature_config.json` (Self-configuring features)

```python
# The API doesn't care if it's LSTM or Linear Regression
model_uri = info['model_uri']  # e.g. "models:/EURUSD_LSTM@champion"
model = mlflow.pyfunc.load_model(model_uri)
```

---

## 5. The Infrastructure ☁️

Why not just run it on a laptop? **Persistence.**

*   🖥️ **EC2**: The Compute (Runs the Server).
*   🗄️ **S3**: The Warehouse (Stores Models, Plots, Artifacts).
*   🗃️ **RDS**: The Librarian (PostgreSQL stores Metadata & Aliases).

*(RDS is crucial: It prevents database locks when multiple workers train at once.)*

---

## 6. The "Utils" Scripts 🛠️

Things break. We built tools to fix them.

*   `tests/inspect_arima_runs.py` 🧐: Checks specifically for ARIMA runs that might be mismatched.
*   `tests/fix_arima_mismatch.py` 🩹: **The Scalpel.** Detects ARIMA models trained on old data sizes and deletes them.
*   `clean_mlflow.py` 🧹: **The Nuclear Option.** Wipes the registry clean.
*   `fix_mlflow_state.py` 🚑: **The Undo Button.** Restores deleted experiments.

---

## ⚠️ Critical Rule: Version Consistency

**MLflow v3.8.1** everywhere.
1.  Local Dev
2.  EC2 Server
3.  Flask App
4.  Retraining Worker

*Mismatch = Protocol Error.*
