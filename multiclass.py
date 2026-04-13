"""
multiclass.py  (OPTIONAL UPGRADE)
----------------------------------
Replaces the two binary classifiers with a single multi-class model
that predicts finishing position P1–P20 directly.

Win probability  = P(class == 1)
Podium probability = P(class == 1) + P(class == 2) + P(class == 3)

This gives more calibrated probabilities than two separate binary models
because the model learns the full competitive distribution across all 20 positions.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


FEATURES = [
    "grid_pos",
    "quali_pos",
    "avg_finish_last3",
    "avg_quali_last3",
    "track_temp",
    "is_wet",
    "driver_id",
]
MODEL_PATH = "model/multiclass_model.pkl"


def train_multiclass(features_path: str = "data/features.csv", test_size: float = 0.2):
    """
    Train a single multi-class Random Forest to predict finishing position (1–20).

    Evaluation uses Mean Absolute Error (MAE) on position —
    e.g. MAE of 2.5 means predictions are off by ~2.5 places on average.
    """
    df = pd.read_csv(features_path)

    # Target is the actual finishing position (integer 1–20)
    df["finish_pos"] = df["finish_pos"].astype(int).clip(1, 20)

    split = int(len(df) * (1 - test_size))
    X_train = df[FEATURES].iloc[:split]
    X_test  = df[FEATURES].iloc[split:]
    y_train = df["finish_pos"].iloc[:split]
    y_test  = df["finish_pos"].iloc[split:]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ Multi-class model trained | MAE: {mae:.2f} positions")

    # Position accuracy within ±1 and ±3
    within_1 = np.mean(np.abs(y_pred - y_test) <= 1)
    within_3 = np.mean(np.abs(y_pred - y_test) <= 3)
    print(f"   Correct within ±1 place : {within_1:.1%}")
    print(f"   Correct within ±3 places: {within_3:.1%}")

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Multi-class model saved → {MODEL_PATH}")

    return model


def predict_multiclass(
    qualifying_order: list[dict],
    track_temp: float = 30.0,
    is_wet: bool = False,
    features_path: str = "data/features.csv",
) -> pd.DataFrame:
    """
    Predict finishing positions and derive win/podium probabilities
    from the multi-class model's class probability distribution.

    Win prob    = P(position == 1)
    Podium prob = P(position == 1) + P(position == 2) + P(position == 3)
    """
    from f1predictor.features import get_driver_encoding, get_driver_averages

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    driver_encoding = get_driver_encoding(features_path)
    driver_avgs = get_driver_averages(features_path)

    df = pd.DataFrame(qualifying_order)
    df["avg_finish_last3"] = df["driver"].map(driver_avgs["avg_finish_last3"]).fillna(10.0)
    df["avg_quali_last3"]  = df["driver"].map(driver_avgs["avg_quali_last3"]).fillna(10.0)
    df["track_temp"] = track_temp
    df["is_wet"] = int(is_wet)
    df["driver_id"] = df["driver"].map(driver_encoding).fillna(-1).astype(int)

    proba = model.predict_proba(df[FEATURES])       # shape: (n_drivers, n_classes)
    classes = model.classes_                         # array of position labels [1, 2, ..., 20]

    def _class_prob(pos: int) -> np.ndarray:
        """Return probability of a specific position for all drivers."""
        if pos in classes:
            idx = list(classes).index(pos)
            return proba[:, idx]
        return np.zeros(len(df))

    df["predicted_pos"] = model.predict(df[FEATURES])
    df["win_prob"]    = (_class_prob(1) * 100).round(1)
    df["podium_prob"] = (
        (_class_prob(1) + _class_prob(2) + _class_prob(3)) * 100
    ).round(1)

    return (
        df[["driver", "grid_pos", "predicted_pos", "win_prob", "podium_prob"]]
        .sort_values("win_prob", ascending=False)
        .reset_index(drop=True)
    )


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: train
    train_multiclass()

    # Step 2: predict
    qualifying = [
        {"driver": "VER", "grid_pos": 1, "quali_pos": 1},
        {"driver": "NOR", "grid_pos": 2, "quali_pos": 2},
        {"driver": "SAI", "grid_pos": 3, "quali_pos": 3},
        {"driver": "HAM", "grid_pos": 4, "quali_pos": 4},
        {"driver": "RUS", "grid_pos": 5, "quali_pos": 5},
    ]

    results = predict_multiclass(qualifying, track_temp=30, is_wet=False)
    print("\n── Multi-class Predictions ──")
    print(results.to_string(index=False))
