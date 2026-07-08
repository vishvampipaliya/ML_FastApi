# Insurance Premium Prediction API

A production-grade ML FastAPI project (Random Forest model → FastAPI → validated JSON API),
built from the course PDF's "Deploying Machine Learning Models with FastAPI" project.

## Project Structure
```
insurance_premium_prediction/
├── app.py                        # FastAPI application (routes)
├── train_model.py                # Trains the model, produces model/model.pkl
├── requirements.txt              # Python dependencies
├── model/
│   ├── model.pkl                 # Trained sklearn Pipeline (already generated)
│   └── predict.py                # Model loading + prediction logic
├── schema/
│   ├── user_input.py             # Pydantic input schema + feature engineering
│   └── prediction_response.py    # Pydantic response schema
└── config/
    └── city_tier.py              # City tier classification data
```

## 1. Setup in VS Code

**VS Code extensions to install** (Extensions panel, `Ctrl+Shift+X` / `Cmd+Shift+X`):
- **Python** (Microsoft) — `ms-python.python`
- **Pylance** (Microsoft) — `ms-python.vscode-pylance`
- *(optional)* **Jupyter** — if you want to explore the model interactively
- *(optional)* **Docker** — if you plan to containerize later

**Python packages to install** (these are the "models"/libraries the project needs —
run in VS Code's integrated terminal, inside your project folder):

```bash
python -m venv myenv
# Windows:
myenv\Scripts\activate
# macOS/Linux:
source myenv/bin/activate

pip install -r requirements.txt
```

This installs: `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pandas`, `numpy`.

In VS Code, select this virtual environment as your interpreter:
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose `myenv`.

## 2. (Re)train the model — already done, model.pkl is included

The original course dataset (`insurance.csv`) wasn't inside the PDF, so
`train_model.py` generates a synthetic dataset using the same feature-engineering
rules (BMI, age group, lifestyle risk, city tier) and trains a `RandomForestClassifier`
inside an sklearn `Pipeline`. A trained `model/model.pkl` is already included, but you
can regenerate it any time:

```bash
python train_model.py
```

Swap in your real `insurance.csv` inside that script if you have the actual dataset.

## 3. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 4. Example request

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

## Endpoints

| Method | Path        | Purpose                          |
|--------|-------------|-----------------------------------|
| GET    | `/`         | Home — confirms API is running   |
| GET    | `/health`   | Health check (for load balancers)|
| POST   | `/predict`  | Returns premium category + confidence scores |