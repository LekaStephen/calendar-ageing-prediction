# Li-ion Battery Calendar Aging Prediction

GitHub-ready Python/Streamlit rebuild foundation based on the uploaded project report.

## Scope
- Inputs: storage temperature (°C), state of charge (SOC, %), storage duration (hours).
- Target: capacity loss relative to initial measured capacity.
- Candidate: RBF Support Vector Regression (SVR); benchmark: Linear Regression.
- Validation: nested grouped cross-validation by storage-condition group.
- Interface: Streamlit; data/model tools: pandas, NumPy, scikit-learn, Matplotlib, Joblib.

The report describes 595 reconstructed observations across 17 storage-condition groups. These are not independent physical-cell trajectories. This demonstrator must not claim validated performance on unseen physical cells or act as a battery-safety assessment.

## Install and run
```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
streamlit run app.py
```

## Dataset
Obtain the authentic Naumann (2021) workbook and place it in `data/raw/`:
Naumann, M. (2021). *Data for: Analysis and modeling of calendar aging of a commercial LiFePO4/graphite cell* (Version 1). Mendeley Data. https://doi.org/10.17632/kxh42bfgtj.1

Do not invent or synthesize replacement observations. Check the dataset reuse terms before committing the workbook to a public repository.

## Rebuild status
This starter includes application structure, input validation, a prediction-service interface, basic tests, and GitHub Actions. It intentionally contains no fabricated data or trained model. The report's input limits are temperature 0–60 °C, SOC 0–100%, and duration 0–21,241 hours. Duration is converted to days for inference. The authentic dataset must be audited and the model trained before prediction is enabled.
