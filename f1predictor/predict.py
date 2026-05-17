"""
predict.py
----------
Predicts win probability (win_model) and podium probability (podium_model)
for each driver in an upcoming race.

Supports:
  - Manual qualifying input (list of dicts)
  - Auto qualifying fetch from FastF1 API
"""

import fastf1
import pandas as pd

from f1predictor.train import load_models, FEATURES
from f1predictor.features import get_driver_averages, get_circuit_podium_rates, load_encoders


MEDAL = {0: "🏆", 1: "🥈", 2: "🥉"}


# ── Auto qualifying fetch ──────────────────────────────────────────────────────

def fetch_qualifying(year: int, round_num: int) -> list[dict]:
    """
    Fetch real qualifying results from FastF1 for a given race weekend.
    Returns a list of dicts with driver, grid_pos, quali_pos, team.
    """
    print(f"📡 Fetching qualifying data for {year} Round {round_num}...")

    fastf1.Cache.enable_cache("f1_cache")
    quali = fastf1.get_session(year, round_num, "Q")
    quali.load(telemetry=False, weather=False, messages=False)

    results = quali.results[["Abbreviation", "Position", "TeamName"]].copy()
    results = results.dropna(subset=["Position"])
    results = results.sort_values("Position")

    qualifying = []
    for i, (_, row) in enumerate(results.iterrows(), start=1):
        qualifying.append({
            "driver":    row["Abbreviation"],
            "grid_pos":  i,
            "quali_pos": int(row["Position"]),
            "team":      row.get("TeamName", "Unknown"),
        })

    if not qualifying:
        raise ValueError(
            f"FastF1 returned 0 drivers for {year} Round {round_num}. "
            f"Qualifying may not be indexed yet — use --manual instead."
        )

    print(f"✅ Fetched {len(qualifying)} drivers from qualifying")
    return qualifying


# ── Core prediction function ───────────────────────────────────────────────────

def predict_race(
    qualifying_order: list[dict],
    track_temp: float = 30.0,
    is_wet: bool = False,
    circuit_name: str = None,
    year: int = 2026,
    podium_model_path: str = "model/podium_model.pkl",
    win_model_path:    str = "model/win_model.pkl",
    threshold_path:    str = "model/thresholds.pkl",
    features_path:     str = "data/features.csv",
    encoder_path:      str = "data/encoders.pkl",
) -> pd.DataFrame:
    """
    Predict win and podium probabilities for each driver.

    Args:
        qualifying_order: list of dicts — required keys: driver, grid_pos, quali_pos
                          optional key: team (improves team_id encoding)
        track_temp:       track temperature in °C
        is_wet:           True if wet race expected
        circuit_name:     circuit name for podium rate lookup (partial match ok)
        year:             season year — used to compute reg_era
    """
    podium_model, win_model, thresholds = load_models(
        podium_model_path, win_model_path, threshold_path
    )

    encoders      = load_encoders(encoder_path)
    driver_enc    = encoders["driver"]
    team_enc      = encoders["team"]
    driver_avgs   = get_driver_averages(features_path)
    circuit_rates = get_circuit_podium_rates(features_path)

    df = pd.DataFrame(qualifying_order)

    # ── Historical form ────────────────────────────────────────────────────────
    df["avg_finish_last3"] = (
        df["driver"].map(driver_avgs["avg_finish_last3"]).fillna(10.0)
    )
    df["avg_quali_last3"] = (
        df["driver"].map(driver_avgs["avg_quali_last3"]).fillna(10.0)
    )
    df["cumulative_podiums"] = (
        df["driver"].map(driver_avgs["cumulative_podiums"]).fillna(0.0)
    )

    # ── Circuit podium rate ────────────────────────────────────────────────────
    if circuit_name:
        subset = circuit_rates[
            circuit_rates["circuit"].str.contains(circuit_name, case=False, na=False)
        ]
        rate_map = subset.set_index("driver")["circuit_podium_rate"].to_dict()
        df["circuit_podium_rate"] = df["driver"].map(rate_map).fillna(0.0)
    else:
        df["circuit_podium_rate"] = 0.0

    # ── Race conditions ────────────────────────────────────────────────────────
    df["track_temp"] = track_temp
    df["is_wet"]     = int(is_wet)
    df["reg_era"]    = 0 if year <= 2021 else (1 if year <= 2025 else 2)

    # ── Encodings ──────────────────────────────────────────────────────────────
    df["driver_id"] = driver_enc.transform(df["driver"].astype(str).tolist())

    if "team" in df.columns:
        df["team_id"] = team_enc.transform(df["team"].astype(str).tolist())
    else:
        team_map = driver_avgs["team_id"].to_dict() if "team_id" in driver_avgs.columns else {}
        df["team_id"] = df["driver"].map(team_map).fillna(-1).astype(int)

    # ── Validate ───────────────────────────────────────────────────────────────
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features before prediction: {missing}")

    X = df[FEATURES]

    # ── Predict ────────────────────────────────────────────────────────────────
    df["win_prob"]    = (win_model.predict_proba(X)[:, 1] * 100).round(1)
    df["podium_prob"] = (podium_model.predict_proba(X)[:, 1] * 100).round(1)

    df = df.sort_values("win_prob", ascending=False).reset_index(drop=True)

    return df[["driver", "grid_pos", "win_prob", "podium_prob"]]


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_predictions(df: pd.DataFrame):
    print(f"\n{'─'*62}")
    print(f"  {'':4} {'Driver':<8} {'Grid':>4}   {'Win%':>6}   {'Podium%':>8}   Bar")
    print(f"{'─'*62}")

    for i, row in df.iterrows():
        medal = MEDAL.get(i, "  ")
        grid  = f"P{int(row['grid_pos']):02d}"
        bar   = "▓" * int(row["win_prob"] / 3)
        print(
            f"  {medal}  {row['driver']:<8} {grid}   "
            f"{row['win_prob']:>5.1f}%   {row['podium_prob']:>6.1f}%   {bar}"
        )

    print(f"{'─'*62}")
    print("  Ranked by Win Probability  |  Podium% shown as context\n")