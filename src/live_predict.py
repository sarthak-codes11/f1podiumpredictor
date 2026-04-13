"""
live_predict.py
---------------
Re-runs the win + podium models each lap using in-race position data
to update predictions as the race evolves.
"""

import pandas as pd
from f1predictor.train import load_models, FEATURES
from f1predictor.features import get_driver_encoding


def predict_for_lap(
    lap_positions: dict,        # {driver: {x, y, position, compound}}
    all_lap_history: dict,      # {lap_num: {driver: {position, ...}}}
    current_lap: int,
    track_temp: float = 30.0,
    is_wet: bool = False,
    features_path: str = "data/features.csv",
) -> dict:
    """
    Re-predict win and podium probabilities using in-race rolling form.

    Instead of historical season averages, uses the last 3 laps'
    positions from the current race as the rolling form signal.

    Returns:
        {driver: {win_prob: float, podium_prob: float}}
    """
    podium_model, win_model = load_models()
    driver_encoding = get_driver_encoding(features_path)

    rows = []
    for driver, data in lap_positions.items():
        # Rolling avg position over last 3 laps (in-race)
        recent_positions = []
        for past_lap in range(max(1, current_lap - 3), current_lap):
            past_data = all_lap_history.get(past_lap, {}).get(driver)
            if past_data:
                recent_positions.append(past_data["position"])

        avg_recent = sum(recent_positions) / len(recent_positions) if recent_positions else 10.0

        rows.append({
            "driver": driver,
            "grid_pos": data.get("position", 10),   # use current race pos as proxy
            "quali_pos": data.get("position", 10),
            "avg_finish_last3": avg_recent,
            "avg_quali_last3": avg_recent,
            "track_temp": track_temp,
            "is_wet": int(is_wet),
            "driver_id": driver_encoding.get(driver, -1),
        })

    df = pd.DataFrame(rows)
    X = df[FEATURES]

    win_probs = win_model.predict_proba(X)[:, 1] * 100
    podium_probs = podium_model.predict_proba(X)[:, 1] * 100

    return {
        row["driver"]: {
            "win_prob": round(float(win_probs[i]), 1),
            "podium_prob": round(float(podium_probs[i]), 1),
        }
        for i, row in enumerate(rows)
    }