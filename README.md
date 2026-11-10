# Li-ion Battery Calendar Ageing Predictor

Academic Streamlit application using the Naumann et al. calendar-ageing dataset only. Oxford datasets are excluded.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

The interface contains four sections: Prediction, Data exploration, Model performance, and About & limitations. Prediction inputs are storage temperature (°C), SOC (%), and duration (hours or days). Duration is converted to days internally.

## Train and save the application model

```bash
python src/train_model.py
```

This reconstructs the 595 condition-time observations from `data/source/DischargeCapacity.xlsx`, performs grouped inner hyperparameter selection (3-fold GroupKFold, MAE), and fits the selected scaled RBF-SVR pipeline on all 17 condition groups for application inference. This full-data fit is not the validation estimate. Use the separately reproduced nested grouped out-of-fold metrics in `reports/nested_grouped_summary_3_4_5fold.csv` for model evaluation.

## Run tests

```bash
pytest -q
```

## Important scope limitations

- Inputs: storage time, temperature, SOC. Time is converted to days internally.
- Supported marginal ranges: 0–21,241 h, 0–60 °C, and 0–100% SOC. Not every combination in these ranges was experimentally observed; the UI flags unrepresented combinations.
- Data grain is condition-level, not physical-cell-level.
- No Oxford data, EIS, or DVA predictors are used. Resistance is not a primary predictor.
- The application returns a point estimate and does not calculate prediction intervals.
- The model has not been externally validated and is not a certified battery safety or battery-management system.

## QA status

See `reports/stage3_application_qa.txt`. In the development environment used to prepare this package, Python compilation passed and all six pytest tests passed. Streamlit was not installed in that environment, so the server launch and browser UI were not verified there. Install requirements and run `streamlit run app.py` locally for the next acceptance test.
