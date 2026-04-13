"""
features.py
-----------
Turns raw race data into ML-ready features.
Adds rolling averages, flags, and encodings.
"""

import os
import pandas as pd


def build_features(input_path: str = "data/raw_data.csv",
                   output_path: str = "data/features.csv") -> pd.DataFrame:
    """
    Load raw data, engineer features, and save result.

    Features created:
        - on_podium:        target variable (finish_pos <= 3)
        - avg_finish_last3: rolling avg finish pos over last 3 races
        - avg_quali_last3:  rolling avg quali pos over last 3 races
        - is_wet:           binary flag for rainfall > 0
        - driver_id:        numeric encoding of driver abbreviation

    Args:
        input_path:  path to raw_data.csv
        output_path: path to save features.csv

    Returns:
        DataFrame with engineered features
    """
    df = pd.read_csv(input_path)
    print(f"📂 Loaded {len(df)} rows from {input_path}")

    # --- Drop rows with missing critical values ---
    before = len(df)
    df = df.dropna(subset=["driver", "grid_pos", "quali_pos", "finish_pos"])
    print(f"🧹 Dropped {before - len(df)} rows with missing values → {len(df)} rows remain")

    # --- Target variable ---
    df["on_podium"] = (df["finish_pos"] <= 3).astype(int)
    print(f"🏆 Podium rows: {df['on_podium'].sum()} / {len(df)}")

    # --- Sort correctly before rolling calculations ---
    df = df.sort_values(["driver", "year", "round"]).reset_index(drop=True)

    # --- Rolling avg finish position (last 3 races, shift(1) = exclude current race) ---
    df["avg_finish_last3"] = (
        df.groupby("driver")["finish_pos"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- Rolling avg qualifying position (last 3 races) ---
    df["avg_quali_last3"] = (
        df.groupby("driver")["quali_pos"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- Wet race binary flag ---
    df["is_wet"] = (df["rainfall"] > 0).astype(int)

    # --- Driver label encoding ---
    df["driver_id"] = df["driver"].astype("category").cat.codes

    # --- Drop rows where rolling features are NaN (first race of career) ---
    before = len(df)
    df = df.dropna(subset=["avg_finish_last3", "avg_quali_last3"])
    print(f"🧹 Dropped {before - len(df)} rows with NaN rolling features → {len(df)} rows remain")

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Features saved to {output_path}")

    return df


def get_driver_encoding(features_path: str = "data/features.csv") -> dict:
    """
    Returns a dict mapping driver abbreviation → driver_id integer.
    Used during prediction to encode new drivers consistently.
    """
    df = pd.read_csv(features_path)
    return df[["driver", "driver_id"]].drop_duplicates().set_index("driver")["driver_id"].to_dict()


def get_driver_averages(features_path: str = "data/features.csv") -> pd.DataFrame:
    """
    Returns a DataFrame with each driver's mean rolling features.
    Used to fill in form stats when predicting a future race.
    """
    df = pd.read_csv(features_path)
    return df.groupby("driver")[["avg_finish_last3", "avg_quali_last3"]].mean()


if __name__ == "__main__":
    build_features()
