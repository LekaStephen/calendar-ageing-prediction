# Stage 6 — Runtime setup, GitHub CI and deployment

## Important

The trained model is already supplied at `models/calendar_ageing_svr.joblib`. Do not run
`python src/train_model.py` as part of normal application startup or deployment. That
script is for intentional model rebuilding only. Keep the model and its metadata together.

## Local setup (Windows PowerShell)

Use Python 3.11 for the documented deployment environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

If PowerShell blocks activation, run the commands from Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Check all four
tabs: Prediction, Data exploration, Model performance, and About & limitations.

## GitHub repository setup

From the project root:

```bash
git init
git add .
git commit -m "Add verified calendar ageing prediction application"
git branch -M main
git remote add origin https://github.com/LekaStephen/calendar-ageing-prediction.git
git push -u origin main
```

If the repository already has a remote, inspect it with `git remote -v` and do not add a
duplicate. If Git reports unrelated existing history, stop and reconcile the histories
rather than force-pushing.

The workflow at `.github/workflows/tests.yml` runs on pushes and pull requests. It installs
`requirements.txt`, compiles the Python files and executes `pytest -q` on GitHub-hosted
Ubuntu with Python 3.11.

## Streamlit Community Cloud deployment

1. Push the complete project to GitHub, including `app.py`, `requirements.txt`, `src/`,
   `models/`, `data/processed/`, and `reports/`.
2. In Streamlit Community Cloud, create an app from the repository
   `LekaStephen/calendar-ageing-prediction`.
3. Select branch `main` and main file path `app.py`.
4. Deploy, then open the app URL and perform the acceptance checks below.

Do not put credentials, tokens, or private keys in the repository. The app currently
requires no secrets. Confirm repository size and hosting policy before publishing the
model and source workbook publicly.

## Browser acceptance checklist

- App opens without an exception and displays the research-use warning.
- Default prediction (25 °C, 50% SOC, 8,760 hours) displays approximately 3.08% loss
  and 96.92% retention.
- Change temperature, SOC and duration; results update after clicking the prediction button.
- Switch between hours and days and verify equivalent durations give equivalent outputs.
- Inspect a represented condition and an unrepresented temperature/SOC pair; the latter
  should show the application's warning.
- Data exploration tab displays condition trajectories and a table.
- Model performance tab displays grouped validation metrics.
- About & limitations tab displays dataset, target definition, model details and limitations.
- Test a narrow mobile viewport and keyboard navigation.

Record the deployment URL, date, browser, test outcome and any errors in the QA report.
A successful GitHub Actions run is automated code/test evidence; it does not replace this
manual browser acceptance test.
