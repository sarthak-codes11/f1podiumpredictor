"""
features.py
-----------
Turns raw race data into ML-ready features.
Adds rolling averages, flags, encodings, and circuit-specific stats.
"""

import os
import pickle
import numpy as np
import pandas as pd
from f1predictor.encoders import SafeLabelEncoder


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(
    input_path: str = "data/raw_data.csv",
    output_path: str = "data/features.csv",
    encoder_path: str = "data/encoders.pkl",
) -> pd.DataFrame:

    df = pd.read_csv(input_path)
    print(f"📂 Loaded {len(df)} rows from {input_path}")

    # ── 1. Drop rows with missing critical values ──────────────────────────────
    before = len(df)
    df = df.dropna(subset=["driver", "grid_pos", "quali_pos", "finish_pos"])
    print(f"🧹 Dropped {before - len(df)} rows with missing values → {len(df)} rows remain")

    # ── 2. Sort before any time-series operations ──────────────────────────────
    df = df.sort_values(["driver", "year", "round"]).reset_index(drop=True)

    # ── 3. Target variables ────────────────────────────────────────────────────
    df["on_podium"]   = (df["finish_pos"] <= 3).astype(int)
    df["race_winner"] = (df["finish_pos"] == 1).astype(int)
    print(f"🏆 Podium rows:  {df['on_podium'].sum()} / {len(df)}")
    print(f"🥇 Winner rows:  {df['race_winner'].sum()} / {len(df)}")

    # ── 4. Rolling avg finish position (last 3 races) ─────────────────────────
    df["avg_finish_last3"] = (
        df.groupby("driver")["finish_pos"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # ── 5. Rolling avg qualifying position (last 3 races) ─────────────────────
    df["avg_quali_last3"] = (
        df.groupby("driver")["quali_pos"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # ── 6. Wet race flag ───────────────────────────────────────────────────────
    if "is_wet" not in df.columns:
        df["is_wet"] = (df["rainfall"] > 0.1).astype(int)
    else:
        df["is_wet"] = df["is_wet"].fillna(0).astype(int)

    # ── 7. Regulation era ──────────────────────────────────────────────────────
    if "reg_era" not in df.columns:
        df["reg_era"] = df["year"].apply(
            lambda y: 0 if y <= 2021 else (1 if y <= 2025 else 2)
        )
    else:
        df["reg_era"] = df["reg_era"].fillna(1).astype(int)

    # ── 8. Circuit podium rate ─────────────────────────────────────────────────
    df["circuit_podium_rate"] = (
        df.groupby(["driver", "circuit"])["on_podium"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0.0)
    )

    # ── 9. Cumulative podiums this season ──────────────────────────────────────
    df["cumulative_podiums"] = (
        df.groupby(["driver", "year"])["on_podium"]
        .transform(lambda x: x.shift(1).cumsum())
        .fillna(0)
    )

    # ── 10. Sample weights ─────────────────────────────────────────────────────
    weight_map = {2022: 1.0, 2023: 1.5, 2024: 2.5, 2025: 3.5, 2026: 5.0}
    df["sample_weight"] = df["year"].map(weight_map).fillna(1.0)

    # ── 11. Driver & team encoding ─────────────────────────────────────────────
    driver_enc = SafeLabelEncoder()
    team_enc   = SafeLabelEncoder()

    df["team"] = df["team"].fillna("Unknown") if "team" in df.columns else "Unknown"

    df["driver_id"] = driver_enc.fit_transform(df["driver"].astype(str).tolist())
    df["team_id"]   = team_enc.fit_transform(df["team"].astype(str).tolist())

    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    with open(encoder_path, "wb") as f:
        pickle.dump({"driver": driver_enc, "team": team_enc}, f)
    print(f"💾 Encoders saved to {encoder_path}")

    # ── 12. Drop NaN rolling rows ──────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["avg_finish_last3", "avg_quali_last3"])
    print(f"🧹 Dropped {before - len(df)} rows with NaN rolling features → {len(df)} rows remain")

    # ── 13. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Features saved to {output_path} ({len(df)} rows)")

    return df


# ── Helper functions ───────────────────────────────────────────────────────────

def get_driver_averages(features_path: str = "data/features.csv") -> pd.DataFrame:
    """Most recent rolling features per driver — used at prediction time."""
    df = pd.read_csv(features_path)
    df = df.sort_values(["driver", "year", "round"])
    cols = ["avg_finish_last3", "avg_quali_last3", "cumulative_podiums", "team_id"]
    return df.groupby("driver").last()[cols]


def get_circuit_podium_rates(features_path: str = "data/features.csv") -> pd.DataFrame:
    """Most recent circuit_podium_rate per (driver, circuit) — used at prediction time."""
    df = pd.read_csv(features_path)
    df = df.sort_values(["driver", "year", "round"])
    return (
        df.groupby(["driver", "circuit"])
        .last()[["circuit_podium_rate"]]
        .reset_index()
    )


def load_encoders(encoder_path: str = "data/encoders.pkl") -> dict:
    """Load fitted SafeLabelEncoders from disk."""
    with open(encoder_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_features()