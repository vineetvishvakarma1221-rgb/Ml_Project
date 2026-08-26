import joblib
import pandas as pd
import numpy as np
from feature_engineering import extract_model_feature
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR/"models"/"best_random_forest.pkl"
feature_path = BASE_DIR/"models"/"feature_names.pkl"
threshold_path = BASE_DIR/"models"/"threshold.pkl"

model = joblib.load(model_path)
feature_names = joblib.load(feature_path)
threshold = joblib.load(threshold_path)

def predict_customer(data_input):
    df = pd.DataFrame([data_input])
    df = df[feature_names]
    probability = model.predict_proba(df)[0][1]
    prediction = int(
        probability >= threshold
    )
    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "fraud_pobability": float(
            round(probability, 4)
        ),
        "prediction": prediction,
        "risk": risk
    }

def batch_predict(df):
    df = df[feature_names].copy()
    probability = model.predict_proba(df)[:, 1]
    prediction = (
        probability >= threshold
    ).astype(int)
    risk = []
    for p in probability:
        if p < 0.30:
            risk.append("Low")
        elif p < 0.60:
            risk.append("Medium")
        else:
            risk.append("High")

    Op = df.copy()
    Op["fraud_pobability"] = probability
    Op["prediction"] = prediction
    Op["risk"] = risk
    return Op

def raw_predict(raw_df):
    # Extract model features from raw consumption data
    feature = extract_model_feature(raw_df)
    # Preserve consumer ID
    consumer_ids = None
    if "CONS_NO" in feature.columns:
        consumer_ids = feature["CONS_NO"].copy()
    # Select only model features
    model_input = feature[
        feature_names
    ].copy()
    # Existing batch prediction logic
    result = batch_predict(model_input)
    # Add consumer ID back to output
    if consumer_ids is not None:
        result.insert(
            0,
            "CONS_NO",
            consumer_ids.values
        )
    return result

# Backend testing
if __name__ == "__main__":
    sample = {
        "std_cons": 0.45,
        "Stability": 0.87,
        "cv": 0.16,
        "longest_missing_streak": 3,
        "high_cons_ratio": 0.42,
        "PAR": 3.5,
        "zero_ratio": 0.01,
        "longest_zero_streak": 1
    }
    print(
        predict_customer(sample)
    )