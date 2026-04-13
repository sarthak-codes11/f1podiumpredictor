"""
predict.py
----------
Predicts win probability (win_model) and podium probability (podium_model)
for each driver in an upcoming race.

Ranking is based ONLY on win probability.
Podium probability is displayed as supporting context.
"""

import pandas as pd
from f1predictor.train import load_models, FEATURES
from f1predictor.features import get_driver_encoding, get_driver_averages


MEDAL = {0: "🏆", 1: "🥈", 2: "🥉"}


def predict_race(
    qualifying_order: list[dict],
    track_temp: float = 30.0,
    is_wet: bool = False,
    podium_model_path: str = "model/podium_model.pkl",
    win_model_path: str = "model/win_model.pkl",
    features_path: str = "data/features.csv",
) -> pd.DataFrame:
    """
    Predict win and podium probabilities for each driver.

    Args:
        qualifying_order: list of dicts with keys: driver, grid_pos, quali_pos
        track_temp:       race track temperature (°C)
        is_wet:           True if wet race expected
        podium_model_path: path to saved podium model
        win_model_path:    path to saved win model
        features_path:     path to features CSV (for encoding + form averages)

    Returns:
        DataFrame sorted by win_prob descending, with podium_prob as context
    """
    podium_model, win_model = load_models(podium_model_path, win_model_path)
    driver_encoding = get_driver_encoding(features_path)
    driver_avgs = get_driver_averages(features_path)

    df = pd.DataFrame(qualifying_order)

    # Fill historical form from dataset averages
    df["avg_finish_last3"] = df["driver"].map(driver_avgs["avg_finish_last3"]).fillna(10.0)
    df["avg_quali_last3"] = df["driver"].map(driver_avgs["avg_quali_last3"]).fillna(10.0)

    # Race conditions
    df["track_temp"] = track_temp
    df["is_wet"] = int(is_wet)

    # Driver encoding — unseen drivers default to -1
    df["driver_id"] = df["driver"].map(driver_encoding).fillna(-1).astype(int)

    X = df[FEATURES]

    # Win probability — used for ranking
    df["win_prob"] = (win_model.predict_proba(X)[:, 1] * 100).round(1)

    # Podium probability — supporting context only
    df["podium_prob"] = (podium_model.predict_proba(X)[:, 1] * 100).round(1)

    # Rank by win probability
    df = df.sort_values("win_prob", ascending=False).reset_index(drop=True)

    return df[["driver", "grid_pos", "win_prob", "podium_prob"]]


def print_predictions(df: pd.DataFrame):
    """
    Pretty-print predictions in the format:

    🏆 NOR   P02   42.1%   76.0%  ▓▓▓▓▓▓▓▓
    🥈 VER   P01   38.5%   67.0%  ▓▓▓▓▓▓▓
    🥉 HAM   P04   34.2%   70.5%  ▓▓▓▓▓▓
    """
    header = f"\n{'─'*58}"
    print(header)
    print(f"  {'':4} {'Driver':<8} {'Grid':>4}   {'Win%':>6}   {'Podium%':>8}   Bar")
    print(f"{'─'*58}")

    for i, row in df.iterrows():
        medal = MEDAL.get(i, "  ")
        grid = f"P{int(row['grid_pos']):02d}"
        bar = "▓" * int(row["win_prob"] / 3)
        print(
            f"  {medal}  {row['driver']:<8} {grid}   "
            f"{row['win_prob']:>5.1f}%   {row['podium_prob']:>6.1f}%   {bar}"
        )

    print(f"{'─'*58}")
    print("  Ranked by Win Probability  |  Podium% shown as context\n")


if __name__ == "__main__":
    # Example: edit this to predict any race
    qualifying = [
        {"driver": "VER", "grid_pos": 1, "quali_pos": 1},
        {"driver": "NOR", "grid_pos": 2, "quali_pos": 2},
        {"driver": "SAI", "grid_pos": 3, "quali_pos": 3},
        {"driver": "HAM", "grid_pos": 4, "quali_pos": 4},
        {"driver": "RUS", "grid_pos": 5, "quali_pos": 5},
        {"driver": "LEC", "grid_pos": 6, "quali_pos": 6},
        {"driver": "PIA", "grid_pos": 7, "quali_pos": 7},
        {"driver": "ALO", "grid_pos": 8, "quali_pos": 8},
        {"driver": "PER", "grid_pos": 9, "quali_pos": 9},
        {"driver": "STR", "grid_pos": 10, "quali_pos": 10},
    ]

    results = predict_race(qualifying, track_temp=30, is_wet=False)
    print_predictions(results)
