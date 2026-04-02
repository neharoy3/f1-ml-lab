import streamlit as st

# Sidebar to choose what you want to do
feature = st.sidebar.selectbox(
    "Choose Feature",
    ["Lap Time Prediction", "Standings Prediction", "Next Feature"]
)

if feature == "Lap Time Prediction":
    # Load lap time model and show inputs
    pass
elif feature == "Standings Prediction":
    # Load standings model and show inputs
    pass