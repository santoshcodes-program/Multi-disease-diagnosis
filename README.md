# Machine Learning-based Multi-Disease Medical Diagnosis using Search Optimization

Clinical Decision Support System (CDSS) that predicts multiple probable diseases from patient symptoms and clinical features, formulated as a search problem in a high-dimensional feature space.

## 1. Problem as Search

- `State`: patient symptom vector + optional clinical attributes.
- `Search space`: all symptom/feature combinations from integrated datasets.
- `Initial state`: user-provided patient input.
- `Search algorithm`: ML learners (Random Forest, SVM, ANN via MLP).
- `Heuristic`: feature importance and class probability scores.
- `Goal state`: ranked disease predictions with probabilities.

The model approximates:

`f(symptoms, clinical_features) -> P(disease | input)`

## 2. Project Structure

```text
.
+-- app/
|   +-- streamlit_app.py
+-- data/
|   +-- raw/                       # put all dataset CSV files here
+-- models/                        # generated model artifacts
+-- reports/                       # generated evaluation outputs
+-- src/
|   +-- config.py
|   +-- data_integration.py
|   +-- feature_search.py
|   +-- predictor.py
|   +-- training.py
|   +-- utils.py
+-- train.py
+-- predict_cli.py
+-- requirements.txt
```

## 3. Dataset Placement

Place all CSV files into `data/raw/`.

Expected files include:

- Dataset 1:
  - `Training.csv`
  - `Testing.csv`
  - `symptom_severity.csv`
  - `description.csv`
  - `precautions.csv`
- Dataset 2:
  - Health symptoms dataset CSV.
- Dataset 3:
  - Diseases/symptoms mapping CSV.
- Dataset 4:
  - Pima diabetes CSV (with `Outcome` or `target`).
- Dataset 5:
  - Heart disease CSV (with `target` or `output`).

Notes:

- The pipeline auto-detects symptom datasets vs diabetes/heart datasets by column patterns.
- `description.csv` and `precautions.csv` are optional metadata enrichments for prediction output.

## 4. Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 5. Train Models

```bash
python train.py
```

Generated outputs:

- Main artifact: `models/multi_disease_model.joblib`
- Optional auxiliary artifacts (if datasets found):
  - `models/diabetes_model.joblib`
  - `models/heart_model.joblib`
- Reports:
  - `reports/model_metrics.csv`
  - `reports/evaluation_details.json`
  - `reports/training_summary.json`

## 6. Run Predictions (CLI)

```bash
python predict_cli.py --symptoms "fever,cough,fatigue,breathlessness" --age 42 --blood-pressure 145 --glucose 130 --top-k 3
```

## 7. Launch Web App

```bash
streamlit run app/streamlit_app.py
```

The app provides:

- Symptom selection.
- Clinical inputs (`age`, `gender`, `blood_pressure`, `glucose`).
- Ranked disease predictions with probabilities.
- Disease descriptions and precautions when metadata files are available.
- Optional diabetes and heart risk modules if their datasets were trained.

## 8. Search Optimization Included

Feature selection reduces the diagnosis search space using:

- Random Forest feature importance.
- Recursive Feature Elimination (RFE).
- Chi-square scoring.

The final feature subset is a union/score-pruned set from these methods.

## 9. Balancing and Evaluation

- Balancing:
  - `SMOTE` (or fallback to `RandomOverSampler` for tiny classes).
- Metrics:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-score (macro)
  - Confusion Matrix
  - ROC-AUC (OvR, when available)

## 10. Important Disclaimer

This system is for educational/research use only and is not a substitute for licensed medical diagnosis.
