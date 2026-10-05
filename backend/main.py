import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Car Price Predictor API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pipeline.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


def calculate_year_range(year):
    bins = [1988, 1993, 1996, 1999, 2004, 2008, 2012, 2016, 2020, 2024, 2030]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = pd.cut([year], bins=bins, labels=labels)[0]
    if pd.isna(result):
        return 10
    return int(result)


def calculate_km_range(kms):
    bins = [0, 30000, 60000, 90000, 120000, 150000, 200000, 500000]
    labels = [1, 2, 3, 4, 5, 6, 7]
    result = pd.cut([kms], bins=bins, labels=labels)[0]
    if pd.isna(result):
        return 7
    return int(result)


def calculate_age_range(age):
    bins = [-1, 2, 5, 10, 15, 20, 25, 30, 36]
    labels = [1, 2, 3, 4, 5, 6, 7, 8]
    result = pd.cut([age], bins=bins, labels=labels)[0]
    if pd.isna(result):
        return 8
    return int(result)


@app.get("/")
def health_check():
    return {"status": "ok", "model": str(MODEL_PATH.name)}


@app.post("/predict")
def predict(data: dict):
    if "Year" in data:
        data["Year_Range"] = int(calculate_year_range(data["Year"]))

    if "KM's driven" in data:
        data["KM's driven_Range"] = int(calculate_km_range(data["KM's driven"]))

    if "Car Age" in data:
        data["Car_Age_Range"] = int(calculate_age_range(data["Car Age"]))

    input_df = pd.DataFrame([data])

    try:
        prediction = model.predict(input_df)
        prediction = prediction.round(0).astype(int)
    except Exception as exc:
        return {"error": str(exc)}

    return {"prediction": prediction.tolist()}