"""
train.py
--------
Trains two Random Forest classifiers:
  - podium_model: predicts finish_pos <= 3
  - win_model:    predicts finish_pos == 1

Both use the same feature set and are saved separately.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


FEATURES = [
    "grid_pos",
    "quali_pos",
    "avg_finish_last3",
    "avg_quali_last3",
    "track_temp",
    "is_wet",
    "driver_id",
]

MODEL_DIR = "model"
PODIUM_MODEL_PATH = f"{MODEL_DIR}/podium_model.pkl"
WIN_MODEL_PATH = f"{MODEL_DIR}/win_model.pkl"


def _train_single(X_train, X_test, y_train, y_test, label: str) -> RandomForestClassifier:
    """Train one RFC and print its evaluation."""
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📈 [{label}] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[f"Not {label}", label]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"🔲 [{label}] Confusion Matrix:")
    print(f"  Correct negatives : {cm[0][0]:3d}  |  False positives : {cm[0][1]:3d}")
    print(f"  Missed positives  : {cm[1][0]:3d}  |  Correct positives: {cm[1][1]:3d}")

    return model


def train(features_path: str = "data/features.csv", test_size: float = 0.2):
    """
    Train both models and save to disk.
    Uses time-ordered split (no shuffle) to avoid data leakage.
    """
    df = pd.read_csv(features_path)
    print(f"📂 Loaded {len(df)} rows")

    split = int(len(df) * (1 - test_size))
    X = df[FEATURES]
    X_train, X_test = X.iloc[:split], X.iloc[split:]

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Podium model ──────────────────────────────────────────────────────────
    y_podium = df["on_podium"]
    y_p_train, y_p_test = y_podium.iloc[:split], y_podium.iloc[split:]
    podium_model = _train_single(X_train, X_test, y_p_train, y_p_test, "Podium")
    with open(PODIUM_MODEL_PATH, "wb") as f:
        pickle.dump(podium_model, f)
    print(f"\n💾 Podium model saved → {PODIUM_MODEL_PATH}")

    # ── Winner model ──────────────────────────────────────────────────────────
    y_win = (df["finish_pos"] == 1).astype(int)
    y_w_train, y_w_test = y_win.iloc[:split], y_win.iloc[split:]
    win_model = _train_single(X_train, X_test, y_w_train, y_w_test, "Winner")
    with open(WIN_MODEL_PATH, "wb") as f:
        pickle.dump(win_model, f)
    print(f"\n💾 Winner model saved → {WIN_MODEL_PATH}")

    # ── Feature importances averaged across both models ───────────────────────
    print("\n🔍 Feature Importances (averaged across both models):")
    avg_imp = [
        (p + w) / 2
        for p, w in zip(podium_model.feature_importances_, win_model.feature_importances_)
    ]
    for feat, imp in sorted(zip(FEATURES, avg_imp), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<22} {bar} {imp:.3f}")

    return podium_model, win_model


def load_models(
    podium_path: str = PODIUM_MODEL_PATH,
    win_path: str = WIN_MODEL_PATH,
):
    """Load both saved models and return (podium_model, win_model)."""
    with open(podium_path, "rb") as f:
        podium_model = pickle.load(f)
    with open(win_path, "rb") as f:
        win_model = pickle.load(f)
    print("✅ Both models loaded")
    return podium_model, win_model


if __name__ == "__main__":
    train()
