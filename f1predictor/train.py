"""
train.py
--------
Trains two Random Forest classifiers:
  - podium_model: predicts finish_pos <= 3
  - win_model:    predicts finish_pos == 1

Both use the same feature set and are saved separately.
Uses time-ordered train/test split to avoid data leakage.
Applies sample weights so recent seasons matter more.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# ── Feature columns ────────────────────────────────────────────────────────────
FEATURES = [
    "grid_pos",
    "quali_pos",
    "avg_finish_last3",
    "avg_quali_last3",
    "track_temp",
    "is_wet",
    "reg_era",
    "driver_id",
    "team_id",
    "circuit_podium_rate",
    "cumulative_podiums",
]

MODEL_DIR         = "model"
PODIUM_MODEL_PATH = f"{MODEL_DIR}/podium_model.pkl"
WIN_MODEL_PATH    = f"{MODEL_DIR}/win_model.pkl"
THRESHOLD_PATH    = f"{MODEL_DIR}/thresholds.pkl"

PODIUM_CLASS_WEIGHT = {0: 1, 1: 6}
WIN_CLASS_WEIGHT    = {0: 1, 1: 55}
PODIUM_THRESHOLD    = 0.40
WIN_THRESHOLD       = 0.30


# ── Helpers ────────────────────────────────────────────────────────────────────

def _time_ordered_split(df: pd.DataFrame, test_year: int = 2026):
    test_df  = df[df["year"] == test_year]
    train_df = df[df["year"] <  test_year]

    if len(test_df) == 0:
        print(f"  ⚠️  No data for {test_year} yet — falling back to last-20%-by-row split")
        split    = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        test_df  = df.iloc[split:]

    print(f"  📊 Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df


def _evaluate(model, X_test, y_test, label: str, threshold: float):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_test, y_prob) if y_test.sum() > 0 else float("nan")

    print(f"\n📈 [{label}] threshold={threshold}  AUC-ROC={auc:.3f}")
    print(classification_report(
        y_test, y_pred,
        target_names=[f"Not {label}", label],
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, y_pred)
    print(f"🔲 [{label}] Confusion Matrix:")
    print(f"  Correct negatives : {cm[0][0]:4d}  |  False positives : {cm[0][1]:4d}")
    print(f"  Missed positives  : {cm[1][0]:4d}  |  Correct positives: {cm[1][1]:4d}")


def _train_single(
    X_train, X_test,
    y_train, y_test,
    sample_weights,
    class_weight: dict,
    label: str,
    threshold: float,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    _evaluate(model, X_test, y_test, label, threshold)
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def train(
    features_path: str = "data/features.csv",
    test_year: int = 2026,
):
    df = pd.read_csv(features_path)
    print(f"📂 Loaded {len(df)} rows from {features_path}")

    # Validate columns
    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
        raise ValueError(
            f"❌ Missing columns in features.csv: {missing_cols}\n"
            f"   Run: python run.py features"
        )

    train_df, test_df = _time_ordered_split(df, test_year)

    X_train = train_df[FEATURES]
    X_test  = test_df[FEATURES]

    weight_map     = {2022: 1.0, 2023: 1.5, 2024: 2.5, 2025: 3.5, 2026: 5.0}
    sample_weights = train_df["year"].map(weight_map).fillna(1.0).values

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Podium model ───────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("🏋️  Training Podium Model (top 3 finish)...")
    podium_model = _train_single(
        X_train, X_test,
        train_df["on_podium"], test_df["on_podium"],
        sample_weights,
        class_weight=PODIUM_CLASS_WEIGHT,
        label="Podium",
        threshold=PODIUM_THRESHOLD,
    )
    with open(PODIUM_MODEL_PATH, "wb") as f:
        pickle.dump(podium_model, f)
    print(f"💾 Podium model saved → {PODIUM_MODEL_PATH}")

    # ── Winner model ───────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("🏋️  Training Winner Model (P1 finish)...")
    win_model = _train_single(
        X_train, X_test,
        train_df["race_winner"], test_df["race_winner"],
        sample_weights,
        class_weight=WIN_CLASS_WEIGHT,
        label="Winner",
        threshold=WIN_THRESHOLD,
    )
    with open(WIN_MODEL_PATH, "wb") as f:
        pickle.dump(win_model, f)
    print(f"💾 Winner model saved → {WIN_MODEL_PATH}")

    # ── Save thresholds ────────────────────────────────────────────────────────
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump({"podium": PODIUM_THRESHOLD, "winner": WIN_THRESHOLD}, f)
    print(f"💾 Thresholds saved → {THRESHOLD_PATH}")

    # ── Feature importances ────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("🔍 Feature Importances (averaged across both models):")
    avg_imp = (podium_model.feature_importances_ + win_model.feature_importances_) / 2
    for feat, imp in sorted(zip(FEATURES, avg_imp), key=lambda x: -x[1]):
        bar = "█" * int(imp * 60)
        print(f"  {feat:<25} {bar} {imp:.4f}")

    return podium_model, win_model


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_models(
    podium_path:    str = PODIUM_MODEL_PATH,
    win_path:       str = WIN_MODEL_PATH,
    threshold_path: str = THRESHOLD_PATH,
):
    """Load both saved models and thresholds."""
    with open(podium_path, "rb") as f:
        podium_model = pickle.load(f)
    with open(win_path, "rb") as f:
        win_model = pickle.load(f)

    thresholds = {"podium": PODIUM_THRESHOLD, "winner": WIN_THRESHOLD}
    if os.path.exists(threshold_path):
        with open(threshold_path, "rb") as f:
            thresholds = pickle.load(f)

    print("✅ Both models and thresholds loaded")
    return podium_model, win_model, thresholds


if __name__ == "__main__":
    train()