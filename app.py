from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from prediction import load_model, predict_capacity_loss

st.set_page_config(
    page_title="Li-ion Calendar Ageing Predictor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONDITIONS = [
    (0, 50), (10, 50), (25, 0), (25, 50), (25, 100),
    (40, 0), (40, 12.5), (40, 25), (40, 37.5), (40, 50),
    (40, 62.5), (40, 75), (40, 87.5), (40, 100),
    (60, 0), (60, 50), (60, 100),
]
DATA_PATH = ROOT / "data/processed/reconstructed_capacity_analysis.csv"
METRICS_PATH = ROOT / "reports/nested_grouped_summary_3_4_5fold.csv"

st.title("🔋 Li-ion Battery Calendar Ageing Predictor")
st.caption("A research demonstrator based on the Naumann et al. calendar-ageing dataset")
st.warning(
    "Academic research tool only. Predictions are point estimates from one experimental dataset; "
    "they are not battery safety, warranty, or deployment decisions."
)

try:
    model, metadata = load_model()
except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.info("Restore the trusted files in the models/ folder (calendar_ageing_svr.joblib and model_metadata.json). Model retraining is not required to run the application.")
    st.stop()

if not DATA_PATH.exists():
    st.error("The processed analysis dataset is missing. Restore it from the supplied project data or run the training script.")
    st.stop()

df = pd.read_csv(DATA_PATH)
required = {"condition_id", "temperature_c", "soc_pct", "storage_time_days", "storage_time_hours", "capacity_loss_pct"}
if not required.issubset(df.columns):
    st.error(f"Processed dataset is missing required columns: {sorted(required - set(df.columns))}")
    st.stop()

with st.sidebar:
    st.header("Prediction inputs")
    temp = st.number_input("Storage temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=1.0)
    soc = st.number_input("State of Charge (SOC, %)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    duration_unit = st.radio("Duration unit", ["Hours", "Days"], horizontal=True)
    duration = st.number_input(
        f"Storage duration ({duration_unit.lower()})",
        min_value=0.0,
        max_value=21241.0 if duration_unit == "Hours" else 21241.0 / 24.0,
        value=8760.0 if duration_unit == "Hours" else 365.0,
        step=24.0 if duration_unit == "Hours" else 1.0,
    )
    hours = duration if duration_unit == "Hours" else duration * 24.0
    predict_clicked = st.button("Estimate capacity loss", type="primary", use_container_width=True)
    st.caption("Supported marginal ranges: 0–60 °C, 0–100% SOC and 0–21,241 hours. Not all combinations were tested.")

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if predict_clicked:
    try:
        st.session_state.last_prediction = {
            "inputs": {"temperature": temp, "soc": soc, "hours": hours},
            "result": predict_capacity_loss(model, temp, soc, hours),
        }
    except ValueError as exc:
        st.session_state.last_prediction = None
        st.error(str(exc))

prediction_tab, exploration_tab, performance_tab, about_tab = st.tabs(
    ["Prediction", "Data exploration", "Model performance", "About & limitations"]
)

with prediction_tab:
    st.subheader("Capacity-loss estimate")
    saved = st.session_state.last_prediction
    if saved is None:
        st.info("Enter the storage conditions in the sidebar and select **Estimate capacity loss**.")
    else:
        inp, result = saved["inputs"], saved["result"]
        is_observed = (inp["temperature"], inp["soc"]) in CONDITIONS
        if not is_observed:
            st.warning(
                "This exact temperature/SOC pair is not one of the 17 experimental conditions. "
                "The model estimate is for an unrepresented combination and has greater uncertainty."
            )
        left, right = st.columns(2)
        left.metric("Estimated capacity loss", f"{result['capacity_loss_pct']:.2f}%")
        right.metric("Estimated capacity retention", f"{result['capacity_retention_pct']:.2f}%")
        st.write(
            f"**Conditions:** {inp['temperature']:g} °C · {inp['soc']:g}% SOC · "
            f"{inp['hours']:,.0f} hours ({inp['hours']/24:.1f} days)"
        )
        st.caption("Capacity retention is calculated as 100% minus predicted capacity loss; it is not a separately trained target.")
        st.info(
            "The output is a point estimate. A prediction interval is not implemented. "
            "Consult the grouped validation results and condition-level limitations before interpreting it."
        )

    st.divider()
    st.subheader("At a glance")
    a, b, c = st.columns(3)
    a.metric("Analysis observations", f"{len(df):,}")
    b.metric("Experimental conditions", f"{df['condition_id'].nunique()}")
    c.metric("Time points per condition", f"{df.groupby('condition_id').size().median():.0f}")

with exploration_tab:
    st.subheader("Observed capacity-loss trajectories")
    st.write(
        "Select an experimental condition to inspect its measured capacity-loss trajectory. "
        "Small negative capacity-loss values are retained because they may reflect measurement variation."
    )
    options = sorted(
        df[["condition_id", "temperature_c", "soc_pct"]].drop_duplicates().to_dict("records"),
        key=lambda x: (x["temperature_c"], x["soc_pct"]),
    )
    labels = [f"{x['temperature_c']:g} °C / {x['soc_pct']:g}% SOC" for x in options]
    selected_label = st.selectbox("Experimental condition", labels)
    selected = options[labels.index(selected_label)]
    one = df[(df.temperature_c == selected["temperature_c"]) & (df.soc_pct == selected["soc_pct"])].sort_values("storage_time_days")
    plot_df = one.set_index("storage_time_days")[["capacity_loss_pct"]]
    st.line_chart(plot_df, y="capacity_loss_pct", x_label="Storage duration (days)", y_label="Capacity loss (%)")
    st.caption(f"{len(one)} observations · {one.storage_time_hours.min():,.0f}–{one.storage_time_hours.max():,.0f} hours")
    st.dataframe(
        one[["storage_time_hours", "capacity_ah", "capacity_loss_pct", "capacity_retention_pct"]]
        if "capacity_ah" in one.columns and "capacity_retention_pct" in one.columns
        else one[["storage_time_hours", "capacity_loss_pct"]],
        use_container_width=True,
        hide_index=True,
    )
    st.divider()
    st.subheader("All-condition overview")
    st.line_chart(df.pivot(index="storage_time_days", columns="condition_id", values="capacity_loss_pct"))

with performance_tab:
    st.subheader("Grouped out-of-fold model evaluation")
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        fold_choice = st.selectbox("Outer grouped cross-validation scheme", ["4-fold", "3-fold", "5-fold"], index=0)
        chosen = metrics[metrics.outer_scheme == fold_choice].copy()
        st.dataframe(chosen, use_container_width=True, hide_index=True)
        svr = chosen[chosen.model.str.contains("SVR", case=False)]
        lr = chosen[chosen.model.str.contains("Linear Regression", case=False)]
        if not svr.empty and not lr.empty:
            s, l = svr.iloc[0], lr.iloc[0]
            x, y, z = st.columns(3)
            x.metric("RBF-SVR MAE", f"{s.MAE_pct_points:.3f} pp")
            y.metric("RBF-SVR RMSE", f"{s.RMSE_pct_points:.3f} pp")
            z.metric("RBF-SVR R²", f"{s.R2:.3f}")
            st.bar_chart(chosen.set_index("model")[["MAE_pct_points", "RMSE_pct_points"]])
            st.caption(
                f"For {fold_choice}, RBF-SVR MAE was {s.MAE_pct_points:.3f} percentage points, "
                f"compared with {l.MAE_pct_points:.3f} for Linear Regression. "
                "The fold schemes are sensitivity analyses, not independent datasets."
            )
        st.warning(
            "Only 17 condition groups are available. Grouped validation prevents repeated time points "
            "from the same condition appearing in both training and held-out folds, but it does not "
            "establish generalisation to independent physical cells, chemistries, or external datasets."
        )
    else:
        st.error("Nested grouped validation summary is missing from the reports folder.")

with about_tab:
    st.subheader("Dataset and model")
    st.markdown(
        "- **Dataset:** Naumann et al. calendar-ageing dataset, Mendeley Data.\n"
        "- **DOI:** [10.17632/kxh42bfgtj.1](https://doi.org/10.17632/kxh42bfgtj.1)\n"
        "- **Model:** RBF Support Vector Regression.\n"
        "- **Benchmark:** Linear Regression.\n"
        "- **Predictors:** storage duration, temperature, and SOC.\n"
        "- **Target:** capacity loss (%)."
    )
    st.subheader("Target definition")
    st.latex(r"\text{Capacity Loss (\%)} = 100\times\frac{Q_0-Q_t}{Q_0}")
    st.subheader("Important limitations")
    for item in [
        "The supplied workbook supports condition-level trajectories; individual physical-cell identifiers are not available for cell-level validation.",
        "Only 17 temperature/SOC conditions are represented. A value within the marginal ranges may still be an untested combination.",
        "The model was evaluated with grouped cross-validation by condition. No independent external dataset is used.",
        "No prediction interval is currently calculated. The displayed estimate should not be interpreted as certainty.",
        "EIS and DVA are supplementary sparse files and are not predictors in this application. Resistance is also excluded from the primary model.",
        "The tool is an academic demonstrator, not a certified battery health, safety, or battery-management system."
    ]:
        st.markdown(f"- {item}")
    st.subheader("Model artefact details")
    st.json({
        "model": metadata.get("model"),
        "features": metadata.get("features"),
        "selected_hyperparameters": metadata.get("selected_hyperparameters"),
        "training_rows": metadata.get("training_rows"),
        "condition_groups": metadata.get("condition_groups"),
        "inner_grouped_cv_folds": metadata.get("inner_grouped_cv_folds"),
        "inner_tuning_MAE_percentage_points": metadata.get("best_inner_grouped_mae_percentage_points"),
    })

st.divider()
st.caption(
    "Research scope: Naumann calendar-ageing dataset only. Analysis grain: condition-level trajectories. "
    "No cell-level or external validation is claimed."
)
