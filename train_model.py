import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from config.city_tier import tier_1_cities, tier_2_cities

np.random.seed(42)

OCCUPATIONS = [
    "retired", "freelancer", "student", "government_job",
    "business_owner", "unemployed", "private_job"
]
ALL_CITIES = tier_1_cities + tier_2_cities + ["Siwan", "Bettiah", "Ambikapur"]


def bmi_of(weight, height):
    return weight / (height ** 2)


def age_group_of(age):
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"


def lifestyle_risk_of(smoker, bmi):
    if smoker and bmi > 30:
        return "high"
    elif smoker or bmi > 27:
        return "medium"
    return "low"


def city_tier_of(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    return 3


def premium_label(age, bmi, smoker, income_lpa, city_tier):
    """Simple rule-based scoring -> Low / Medium / High, used only to
    generate plausible synthetic labels for training."""
    score = 0
    score += 2 if age > 55 else (1 if age > 35 else 0)
    score += 2 if bmi > 30 else (1 if bmi > 25 else 0)
    score += 2 if smoker else 0
    score += 1 if city_tier == 1 else 0
    score -= 1 if income_lpa > 20 else 0

    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    return "Low"


def generate_synthetic_data(n=3000):
    rows = []
    for _ in range(n):
        age = np.random.randint(18, 80)
        weight = round(np.random.normal(70, 15), 1)
        weight = max(35.0, weight)
        height = round(np.random.uniform(1.45, 1.95), 2)
        income_lpa = round(np.random.exponential(10) + 1, 2)
        smoker = bool(np.random.rand() < 0.25)
        city = np.random.choice(ALL_CITIES)
        occupation = np.random.choice(OCCUPATIONS)

        bmi = bmi_of(weight, height)
        ctier = city_tier_of(city)
        label = premium_label(age, bmi, smoker, income_lpa, ctier)

        rows.append({
            "age": age,
            "weight": weight,
            "height": height,
            "income_lpa": income_lpa,
            "smoker": smoker,
            "city": city,
            "occupation": occupation,
            "insurance_premium_category": label
        })
    return pd.DataFrame(rows)


def main():
    print("Generating synthetic training data...")
    df = generate_synthetic_data()

    # ---- Feature engineering (matches schema/user_input.py) ----
    df["bmi"] = df.apply(lambda r: bmi_of(r["weight"], r["height"]), axis=1)
    df["age_group"] = df["age"].apply(age_group_of)
    df["lifestyle_risk"] = df.apply(lambda r: lifestyle_risk_of(r["smoker"], r["bmi"]), axis=1)
    df["city_tier"] = df["city"].apply(city_tier_of)

    feature_cols = ["bmi", "age_group", "lifestyle_risk", "city_tier", "income_lpa", "occupation"]
    X = df[feature_cols]
    y = df["insurance_premium_category"]

    categorical_features = ["age_group", "lifestyle_risk", "occupation", "city_tier"]
    numeric_features = ["bmi", "income_lpa"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    with open("model/model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Saved trained pipeline to model/model.pkl")


if __name__ == "__main__":
    main()