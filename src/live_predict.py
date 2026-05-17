"""
live_predict.py
---------------
Re-runs the win + podium models each lap using in-race position data
to update predictions as the race evolves.
"""

import pandas as pd
from f1predictor.train import load_models, FEATURES
from f1predictor.features import load_encoders, get_driver_averages, get_circuit_podium_rates


def predict_for_lap(
    lap_positions: dict,
    all_lap_history: dict,
    current_lap: int,
    track_temp: float = 30.0,
    is_wet: bool = False,
    circuit_name: str = None,
    year: int = 2026,
    features_path: str = "data/features.csv",
    encoder_path: str = "data/encoders.pkl",
) -> dict:
    """
    Re-predict win and podium probabilities using in-race rolling form.

    Instead of historical season averages, uses the last 3 laps'
    positions from the current race as the rolling form signal.

    Returns:
        {driver: {win_prob: float, podium_prob: float}}
    """
    podium_model, win_model, thresholds = load_models()

    encoders      = load_encoders(encoder_path)
    driver_enc    = encoders["driver"]
    team_enc      = encoders["team"]
    driver_avgs   = get_driver_averages(features_path)
    circuit_rates = get_circuit_podium_rates(features_path)

    # Circuit podium rate lookup
    if circuit_name:
        subset   = circuit_rates[
            circuit_rates["circuit"].str.contains(circuit_name, case=False, na=False)
        ]
        rate_map = subset.set_index("driver")["circuit_podium_rate"].to_dict()
    else:
        rate_map = {}

    rows = []
    for driver, data in lap_positions.items():
        # Rolling avg position over last 3 laps (in-race)
        recent_positions = []
        for past_lap in range(max(1, current_lap - 3), current_lap):
            past_data = all_lap_history.get(past_lap, {}).get(driver)
            if past_data:
                recent_positions.append(past_data["position"])

        avg_recent = (
            sum(recent_positions) / len(recent_positions)
            if recent_positions else 10.0
        )

        rows.append({
            "driver":             driver,
            "grid_pos":           data.get("position", 10),
            "quali_pos":          data.get("position", 10),
            "avg_finish_last3":   avg_recent,
            "avg_quali_last3":    avg_recent,
            "track_temp":         track_temp,
            "is_wet":             int(is_wet),
            "reg_era":            0 if year <= 2021 else (1 if year <= 2025 else 2),
            "circuit_podium_rate": rate_map.get(driver, 0.0),
            "cumulative_podiums": (
                driver_avgs["cumulative_podiums"].get(driver, 0.0)
                if "cumulative_podiums" in driver_avgs.columns else 0.0
            ),
            "team_id": (
                driver_avgs["team_id"].get(driver, -1)
                if "team_id" in driver_avgs.columns else -1
            ),
        })

    df = pd.DataFrame(rows)

    # Encode drivers safely
    df["driver_id"] = driver_enc.transform(df["driver"].astype(str).tolist())

    X = df[FEATURES]

    win_probs    = win_model.predict_proba(X)[:, 1] * 100
    podium_probs = podium_model.predict_proba(X)[:, 1] * 100

    return {
        row["driver"]: {
            "win_prob":    round(float(win_probs[i]), 1),
            "podium_prob": round(float(podium_probs[i]), 1),
        }
        for i, row in enumerate(rows)
    }