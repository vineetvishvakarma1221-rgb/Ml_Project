import pandas as pd
import numpy as np

MODEL_FEATURES = [
    "std_cons","Stability","cv","longest_missing_streak",
    "high_cons_ratio","PAR","zero_ratio","longest_zero_streak"
]

def longest_zero_streak(values):
    max_streak = 0
    current_streak = 0
    for value in values:
        if value == 0:
            current_streak += 1
            max_streak = max(max_streak,current_streak)
        else:
            current_streak = 0
    return max_streak

def longest_missing_streak(values):
    max_streak = 0
    current_streak = 0
    for value in values:
        if pd.isna(value):
            current_streak += 1
            max_streak = max(max_streak,current_streak)
        else:
            current_streak = 0
    return max_streak

def extract_model_feature(raw_df):
    df = raw_df.copy()
    id_candidates = [
        "Consumer_ID","consumer_id","CONS_NO","Consumer ID",
        "Contomer_ID","cons_no","Contomer_id","Consumer"]
    cons_id = None
    for col in id_candidates:
        if col in df.columns:
            cons_id = col
            break

    if cons_id is None:
        df.insert(0,"CONS_NO",
            range(1, len(df) + 1)
        )
        cons_id = "CONS_NO"

    exclude_column = {
        cons_id,"CONS_NO","FLAG","flag","Flag",
        "Target","target","Fraud","fraud","fruad"
    }

    reading_column = [
        col for col in df.columns
        if col not in exclude_column
    ]

    if not reading_column:
        raise ValueError(
            "No consumption reading columns were detected."
        )

    readings = df[reading_column].apply(
        pd.to_numeric,
        errors="coerce"
    )

    total_days = len(reading_column)
    missing_days = readings.isna().sum(axis=1)
    valid_days = total_days - missing_days

    valid_days_safe = valid_days.replace(0,np.nan)

    mean_cons = readings.mean(axis=1)
    std_cons = readings.std(axis=1)
    max_cons = readings.max(axis=1)

    # ZERO CONSUMPTION
    zero_days = readings.eq(0).sum(axis=1)
    zero_ratio = (zero_days /valid_days_safe)
    
    # COEFFICIENT OF VARIATION
    cv = (std_cons /(mean_cons + 1))

    # PEAK TO AVERAGE RATIO
 
    PAR = (max_cons /mean_cons.replace(0,np.nan))
    PAR = PAR.fillna(PAR.median())
  
    # LONGEST ZERO STREAK
    longest_zero = readings.apply(
        longest_zero_streak,
        axis=1
    )

    longest_missing = readings.apply(
        longest_missing_streak,
        axis=1
    )

    high_days = readings.gt(
        mean_cons,
        axis=0
    )

    high_cons_days = high_days.sum(
        axis=1
    )

    high_cons_ratio = (
        high_cons_days /
        valid_days_safe
    )

    # STABILITY
    Stability = (1 /(1 + cv))

    # FINAL MODEL FEATURE DATAFRAME
    feature_df = pd.DataFrame({
        "std_cons": std_cons,
        "Stability": Stability,
        "cv": cv,
        "longest_missing_streak": longest_missing,
        "high_cons_ratio": high_cons_ratio,
        "PAR": PAR,
        "zero_ratio": zero_ratio,
        "longest_zero_streak": longest_zero
    })

    # Replace infinity values
    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )
    # Preserve Consumer ID
    feature_df.insert(
        0,
        "CONS_NO",
        df[cons_id].values
    )
    return feature_df