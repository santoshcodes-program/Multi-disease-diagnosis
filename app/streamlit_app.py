from __future__ import annotations

import os
from pathlib import Path
import sys
import importlib.util

import pandas as pd
import streamlit as st

# Ensure project root is importable when Streamlit runs from app/ path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.predictor import MultiDiseasePredictor, load_optional_aux_predictors
    from src.training import train_pipeline
except ModuleNotFoundError:
    # Absolute fallback: load modules directly from source files.
    def _load_attr(module_file: Path, module_name: str, attr_name: str):
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Could not load module spec: {module_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr_name)

    MultiDiseasePredictor = _load_attr(
        PROJECT_ROOT / "src" / "predictor.py",
        "predictor_fallback",
        "MultiDiseasePredictor",
    )
    load_optional_aux_predictors = _load_attr(
        PROJECT_ROOT / "src" / "predictor.py",
        "predictor_fallback_2",
        "load_optional_aux_predictors",
    )
    train_pipeline = _load_attr(
        PROJECT_ROOT / "src" / "training.py",
        "training_fallback",
        "train_pipeline",
    )


st.set_page_config(page_title="Multi-Disease Diagnosis CDSS", page_icon="CDSS", layout="wide")

st.title("Machine Learning Clinical Decision Support System")
st.caption("Multi-disease prediction with search-optimized feature space")

artifact_path = Path("models") / "multi_disease_model.joblib"
dataset_path = Path("data") / "processed" / "unified_symptom_dataset_132.csv"

if not artifact_path.exists():
    st.warning("Model artifact not found. Running fast first-time training...")
    if not dataset_path.exists():
        st.error(
            "Dataset not found for auto-training. Add `data/processed/unified_symptom_dataset_132.csv` "
            "or run training manually."
        )
        st.stop()
    with st.spinner("Training fast model (first run only)..."):
        train_pipeline(
            dataset_path=dataset_path,
            fast_mode=True,
            fast_top_diseases=100,
            fast_max_rows=15000,
            fast_n_estimators=80,
        )
    st.success("Model created successfully.")

predictor = MultiDiseasePredictor(artifact_path=artifact_path)
aux_predictors = load_optional_aux_predictors()
all_symptoms = predictor.available_symptoms()

with st.sidebar:
    st.header("Patient Profile")
    age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
    gender = st.selectbox("Gender", options=["male", "female", "other"])
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
    top_k = st.slider("Top probable diseases", min_value=1, max_value=10, value=3)

pregnancies, skin_thickness, insulin, bmi, pedigree = 1, 20.0, 80.0, 26.0, 0.5
cp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = 1, 210.0, 0, 1, 150.0, 0, 0.0, 1, 0, 2

if aux_predictors:
    st.subheader("Auxiliary Clinical Inputs")
    if "diabetes" in aux_predictors:
        with st.expander("Diabetes Module Inputs", expanded=False):
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
            skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=120.0, value=20.0, step=1.0)
            insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=26.0, step=0.1)
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5, step=0.01)

    if "heart" in aux_predictors:
        with st.expander("Heart Module Inputs", expanded=False):
            cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=4, value=1, step=1)
            chol = st.number_input("Cholesterol", min_value=0.0, max_value=700.0, value=210.0, step=1.0)
            fbs = st.number_input("Fasting Blood Sugar Flag (fbs)", min_value=0, max_value=1, value=0, step=1)
            restecg = st.number_input("Rest ECG (restecg)", min_value=0, max_value=2, value=1, step=1)
            thalach = st.number_input("Max Heart Rate (thalach)", min_value=0.0, max_value=250.0, value=150.0, step=1.0)
            exang = st.number_input("Exercise Angina (exang)", min_value=0, max_value=1, value=0, step=1)
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope = st.number_input("Slope", min_value=0, max_value=3, value=1, step=1)
            ca = st.number_input("CA", min_value=0, max_value=4, value=0, step=1)
            thal = st.number_input("Thal", min_value=0, max_value=3, value=2, step=1)

st.subheader("Symptoms")
selected_symptoms = st.multiselect("Select symptoms", options=all_symptoms)

clinical_features = {
    "age": age,
    "gender": {"male": 1, "female": 0}.get(gender, 0),
    "blood_pressure": blood_pressure,
    "glucose": glucose,
}

if st.button("Predict Diseases", type="primary"):
    predictions = predictor.predict_top_k(
        symptoms=selected_symptoms,
        clinical_features=clinical_features,
        top_k=top_k,
    )
    if not predictions:
        st.warning("No predictions available.")
    else:
        table_df = pd.DataFrame(
            {
                "Disease": [item["disease"] for item in predictions],
                "Probability (%)": [round(item["probability"] * 100, 2) for item in predictions],
            }
        )
        st.subheader("Predicted Diseases")
        st.dataframe(table_df, use_container_width=True)

        st.subheader("Clinical Notes")
        for item in predictions:
            st.markdown(f"**{item['disease']}** - {item['probability'] * 100:.2f}%")
            if item["description"]:
                st.write(f"Description: {item['description']}")
            if item["precautions"]:
                st.write("Precautions:")
                for precaution in item["precautions"]:
                    st.write(f"- {precaution}")

    if aux_predictors:
        st.subheader("Auxiliary Risk Modules")
        if "diabetes" in aux_predictors:
            diabetes_input = {
                "pregnancies": pregnancies,
                "glucose": glucose,
                "bloodpressure": blood_pressure,
                "skinthickness": skin_thickness,
                "insulin": insulin,
                "bmi": bmi,
                "diabetespedigreefunction": pedigree,
                "age": age,
            }
            diabetes_result = aux_predictors["diabetes"].predict_risk(diabetes_input)
            st.write(f"Diabetes risk: {diabetes_result['risk_probability'] * 100:.2f}%")

        if "heart" in aux_predictors:
            heart_input = {
                "age": age,
                "sex": {"male": 1, "female": 0}.get(gender, 0),
                "trestbps": blood_pressure,
                "chol": chol,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "cp": cp,
                "fbs": fbs,
                "restecg": restecg,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
            heart_result = aux_predictors["heart"].predict_risk(heart_input)
            st.write(f"Heart disease risk: {heart_result['risk_probability'] * 100:.2f}%")
