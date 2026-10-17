import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Load trained model and scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Application interface
# -----------------------------
st.title("Heart Disease Prediction")

st.write(
    "An educational Machine Learning project "
    "built using Python and Scikit-learn."
)

st.info(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)

st.divider()

st.subheader("Enter Patient Information")

# Numerical inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        max_value=300,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0.00,
        max_value=700.00,
        value=200.00
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=0,
        max_value=250,
        value=150
    )

with col2:
    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["N", "Y"]
    )

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict", use_container_width=True):

    # Create input using original feature names
    input_data = pd.DataFrame([{
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope
    }])

    # Apply the same one-hot encoding strategy
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope"
        ],
        drop_first=True,
        dtype=int
    )

    # Match the exact training columns and order
    input_data = input_data.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    # Scale using the previously fitted scaler
    scaled_input = scaler.transform(input_data)

    # Predict using the trained model
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "The model predicted YES, there is a risk of heart disease."
        )
    else:
        st.success(
            "The model predicted NO, there is no significant risk of heart disease."
        )

    st.caption(
        "This is an experimental model prediction, "
        "not a medical diagnosis. Consult a qualified "
        "healthcare professional for medical concerns."
    )