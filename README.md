# Multi-Disease Medical Diagnosis (ML + Search Optimization)

A Machine Learning based Clinical Decision Support System that predicts **top probable diseases** from symptom and clinical input.

The app supports:
- Free-text symptom input (example: `I have cold and feeling sick`)
- Symptom extraction + normalization
- Ranked disease output (top-k, default top-3)
- Probability scores
- Optional diabetes and heart risk modules

## Overview

This project frames diagnosis as a search problem in symptom-feature space:
- `State`: patient symptom vector + clinical features
- `Search space`: all symptom combinations from integrated datasets
- `Search strategy`: trained ML models (RF / SVM / ANN)
- `Heuristic`: feature relevance and predicted class probability
- `Goal`: ranked disease predictions with confidence

## Included Datasets (Integrated)

1. Symptom Disease Prediction Dataset (Kaggle)
2. Health Symptoms and Disease Prediction Dataset (Kaggle)
3. Diseases and Symptoms Dataset (Kaggle)

The integration pipeline normalizes feature names and builds unified symptom-disease training data.

## Project Structure

```text
.
|-- app/
|   |-- streamlit_app.py
|-- data/
|   |-- raw/
|   |-- processed/
|       |-- unified_symptom_dataset_132.csv
|-- models/
|   |-- multi_disease_model.joblib
|-- reports/
|-- src/
|   |-- config.py
|   |-- data_integration.py
|   |-- feature_search.py
|   |-- predictor.py
|   |-- symptom_text.py
|   |-- training.py
|   |-- utils.py
|-- build_unified_dataset.py
|-- train.py
|-- predict_cli.py
|-- requirements.txt
```

## Quick Start (Local)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open: `http://localhost:8501`

## Web App Usage

1. Enter symptoms in plain text:
   - `I have cold and feeling sick`
   - `fever, cough, fatigue, headache`
2. Set patient profile (age, gender, BP, glucose)
3. Click **Predict Diseases**
4. App returns ranked predictions with probabilities

Notes:
- The app maps colloquial phrases to symptom features.
- Unknown words are ignored and shown as unmatched terms.

## Training

### Fast training (recommended for large merged data)

```bash
python train.py --dataset-path data/processed/unified_symptom_dataset_132.csv --fast-mode --fast-top-diseases 100 --fast-max-rows 15000 --fast-n-estimators 80
```

### Full training

```bash
python train.py --dataset-path data/processed/unified_symptom_dataset_132.csv
```

Outputs:
- `models/multi_disease_model.joblib`
- `reports/model_metrics.csv`
- `reports/evaluation_details.json`
- `reports/training_summary.json`

## CLI Prediction

```bash
python predict_cli.py --symptoms "fever,cough,fatigue" --age 42 --blood-pressure 140 --glucose 120 --top-k 3
```

## Deploy (Streamlit Community Cloud)

1. Push repository to GitHub
2. Streamlit Cloud -> New app
3. Set entrypoint: `app/streamlit_app.py`
4. Deploy

Current repo includes:
- `data/processed/unified_symptom_dataset_132.csv`
- lightweight pretrained model `models/multi_disease_model.joblib`

So startup is fast and does not require long training.

## Important Disclaimer

This project is for educational/research use only.
It is **not** a replacement for professional medical diagnosis.
