import streamlit as st

from battery_ageing.config import APP_TITLE, RAW_DATA_DIR, MODEL_PATH
from battery_ageing.prediction import predict_capacity_metrics, ModelArtifactError
from battery_ageing.validation import validate_inputs

st.set_page_config(page_title=APP_TITLE, page_icon="🔋", layout="wide")
st.title("🔋 Machine Learning for Calendar Aging Prediction")
st.caption("Research demonstrator based on the Naumann (2021) calendar-ageing dataset.")
st.warning("Research use only. Not a battery management system, safety assessment, warranty tool, or certified engineering instrument.")

predict_tab, status_tab, about_tab = st.tabs(["Capacity prediction", "Data and model status", "About and limitations"])

with predict_tab:
    a, b, c = st.columns(3)
    temperature = a.number_input("Storage temperature (°C)", 0.0, 60.0, 25.0)
    soc = b.number_input("State of charge (%)", 0.0, 100.0, 50.0)
    duration = c.number_input("Storage duration (hours)", 0.0, 21241.0, 8760.0, step=24.0)
    if st.button("Predict capacity loss", type="primary"):
        try:
            values = validate_inputs(temperature, soc, duration)
            result = predict_capacity_metrics(*values)
            st.metric("Predicted capacity loss", f"{result['capacity_loss_percent']:.4f}%")
            st.metric("Estimated capacity retention", f"{result['capacity_retention_percent']:.4f}%")
        except (ValueError, ModelArtifactError) as exc:
            st.error(str(exc))

with status_tab:
    st.write("Raw-data directory:", str(RAW_DATA_DIR))
    st.write("Raw-data directory exists:", RAW_DATA_DIR.exists())
    st.write("Trained model exists:", MODEL_PATH.exists())
    st.info("Prediction is unavailable until the authentic source workbook is audited and a model is trained.")

with about_tab:
    st.markdown("The model estimates capacity loss from temperature, SOC and storage duration. Repeated measurements within storage-condition groups must be respected during validation. The dataset does not establish independent physical-cell trajectories; unseen-cell performance is not claimed.")
    st.markdown("Predictions are research estimates, not safety assessments. Marginal input checks do not prove that every combination is represented in the training data.")
