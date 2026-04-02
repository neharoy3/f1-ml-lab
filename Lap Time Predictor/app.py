import streamlit as st
import joblib
import pandas as pd
import os

# -------------------------
# Load model + feature columns
# -------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "f1_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="F1 ML Lab", page_icon="🏎️")

st.title("🏎️ F1 Lap Time Predictor")

st.markdown("Predict lap time based on race conditions.")

# Inputs
lap = st.number_input("Lap Number", min_value=1, max_value=70, value=10)
tyre = st.number_input("Tyre Age (laps)", min_value=0, max_value=50, value=5)
stint = st.number_input("Stint Number", min_value=1, max_value=5, value=1)
position = st.number_input("Race Position", min_value=1, max_value=20, value=10)

driver = st.selectbox(
    "Driver",
    ["VER", "HAM", "LEC", "NOR", "RUS", "SAI", "ALO", "PER", "PIA", "GAS"]
)

compound = st.selectbox(
    "Tyre Compound",
    ["SOFT", "MEDIUM", "HARD"]
)

# -------------------------
# Prediction logic
# -------------------------
if st.button("Predict Lap Time"):

    # Create base input
    input_data = {
        "LapNumber": lap,
        "TyreLife": tyre,
        "Stint": stint,
        "Position": position,
    }

    df = pd.DataFrame([input_data])

    # One-hot encode driver + compound
    df[f"Driver_{driver}"] = 1
    df[f"Compound_{compound}"] = 1

    # Add missing columns (IMPORTANT)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[feature_columns]

    # Predict
    prediction = model.predict(df)[0]

    st.success(f"⏱️ Predicted Lap Time: {prediction:.2f} seconds")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("**F1 ML Lab** | Built for learning & experimentation")