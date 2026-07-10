# Insurance Premium Prediction API

A production-grade ML FastAPI project — a trained Random Forest model served behind
a validated, documented REST API. Predicts whether a user's insurance premium will
be **Low**, **Medium**, or **High** based on demographic and lifestyle data.

---

## Project Structure

```
insurance_premium_prediction/
├── app.py                        # FastAPI application (routes)
├── streamlit_app.py              # Streamlit frontend (calls the FastAPI /predict endpoint)
├── train_model.py                # Trains the model, produces model/model.pkl
├── requirements.txt              # Python dependencies
├── README.md
├── model/
│   ├── model.pkl                 # Trained sklearn Pipeline (already generated)
│   └── predict.py                # Model loading + prediction logic
├── schema/
│   ├── user_input.py             # Pydantic input schema + feature engineering
│   └── prediction_response.py    # Pydantic response schema
└── config/
    └── city_tier.py              # City tier classification data
```

---

## 1. Prerequisites

- Python 3.9+ installed
- VS Code with these extensions:
  - **Python** (Microsoft) — `ms-python.python`
  - **Pylance** (Microsoft) — `ms-python.vscode-pylance`

---

## 2. Setup

Open this folder in VS Code so that `app.py` sits at the root (imports like
`from schema.user_input import UserInput` depend on running from here).

Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:

```bash
python -m venv myenv
```

Activate it:

| OS | Command |
|---|---|
| Windows (PowerShell) | `myenv\Scripts\Activate.ps1` |
| Windows (cmd) | `myenv\Scripts\activate.bat` |
| macOS / Linux | `source myenv/bin/activate` |

You should see `(myenv)` in the prompt. Then select it as your interpreter:
`Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `myenv` one.

Install dependencies:

```bash
pip install -r requirements.txt
```

This installs: `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pandas`, `numpy`.

---

## 3. (Re)train the model — optional, model.pkl is already included

The original course dataset (`insurance.csv`) isn't bundled here, so
`train_model.py` generates a synthetic dataset using the same feature-engineering
rules (BMI, age group, lifestyle risk, city tier) and trains a `RandomForestClassifier`
inside an sklearn `Pipeline`.

To retrain (e.g. after swapping in your real dataset):

```bash
python train_model.py
```

---

## 4. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Then open:
- **Swagger UI** (interactive): http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 5. Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "weight": 75.5,
    "height": 1.75,
    "income_lpa": 15.0,
    "smoker": false,
    "city": "Mumbai",
    "occupation": "private_job"
  }'
```

Response:
```json
{
  "response": {
    "predicted_category": "Low",
    "confidence": 0.73,
    "class_probabilities": {"Low": 0.73, "Medium": 0.18, "High": 0.09}
  }
}
```

---

## 6. Run the Streamlit frontend (optional)

`streamlit_app.py` lives at the project root, right next to `app.py` (see the
Project Structure tree above). It gives you a simple UI to test predictions
without curl/Swagger.

Make sure the FastAPI server is already running in one terminal, then in a **second**
terminal (same virtual environment):

```bash
streamlit run streamlit_app.py
```

This opens a browser tab at `http://localhost:8501` with a form for age, weight,
height, income, smoker status, city, and occupation. Clicking **Predict Premium
Category** calls your local `/predict` endpoint and displays the predicted category,
confidence percentage, and a bar chart of class probabilities.

> If deploying the API elsewhere, update `API_URL` at the top of `streamlit_app.py`.

## 7. Endpoints

| Method | Path        | Purpose                                       |
|--------|-------------|------------------------------------------------|
| GET    | `/`         | Home — confirms API is running                |
| GET    | `/health`   | Health check (status, model version, model loaded) |
| POST   | `/predict`  | Returns premium category + confidence scores  |

### `/predict` request fields

| Field       | Type  | Constraints                          |
|-------------|-------|----------------------------------------|
| age         | int   | 0 < age < 120                          |
| weight      | float | > 0 (kg)                               |
| height      | float | 0 < height < 2.5 (meters)              |
| income_lpa  | float | > 0 (Lakhs Per Annum)                  |
| smoker      | bool  | true / false                           |
| city        | str   | any city name (case-insensitive)       |
| occupation  | str   | one of: retired, freelancer, student, government_job, business_owner, unemployed, private_job |

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'schema'` (or `'model'`) | Your terminal isn't in the project root. `cd` into the folder containing `app.py` before running uvicorn. |
| `ModuleNotFoundError: No module named 'fastapi'` | Virtual environment isn't activated, or `pip install -r requirements.txt` wasn't run in it. |
| `422 Unprocessable Entity` on `/predict` | Check the request JSON against the field constraints above — Pydantic rejects out-of-range or missing fields. |
| Port 8000 already in use | Run with a different port: `uvicorn app:app --reload --port 8001` |

---

## 9. Key design points

- **Feature engineering lives in the Pydantic model** (`schema/user_input.py`) via `computed_field` — so the exact same BMI / age_group / lifestyle_risk / city_tier logic used in training is guaranteed to run at inference time.
- **The whole preprocessing + model is one pickled sklearn `Pipeline`** — no separate encoder files to keep in sync.
- **`response_model=PredictionResponse`** on the `/predict` route validates and documents the output shape automatically.
- **`/health`** is designed for load balancers / Kubernetes probes, separate from the human-facing `/`.