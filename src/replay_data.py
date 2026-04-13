"""
replay_data.py
--------------
Loads lap-by-lap car positions from FastF1 for a given session.
Used to animate the track map in the replay tab.
"""

import fastf1
import pandas as pd
import numpy as np


def load_lap_positions(year: int, round_num: int, session_type: str = "R") -> dict:
    """
    Load lap-by-lap car positions using the car's actual position
    at the END of each lap (not midpoint), matched to the track reference line.
    """
    fastf1.Cache.enable_cache("f1_cache")

    session = fastf1.get_session(year, round_num, session_type)
    session.load(telemetry=True, weather=False, messages=False)

    laps = session.laps
    max_lap = int(laps["LapNumber"].max())

    # Build track reference line from fastest lap ONCE
    fastest = session.laps.pick_fastest()
    ref_tel = fastest.get_telemetry()[["X", "Y"]].dropna()
    ref_points = ref_tel.values  # shape (N, 2)

    lap_data = {}

    for lap_num in range(1, max_lap + 1):
        lap_data[lap_num] = {}
        lap_slice = laps[laps["LapNumber"] == lap_num]

        for _, lap_row in lap_slice.iterrows():
            driver = lap_row["Driver"]
            try:
                tel = lap_row.get_telemetry()[["X", "Y", "Distance"]].dropna()
                if tel is None or tel.empty:
                    continue

                # Use the point at 50% distance through the lap
                # (end of lap often has GPS glitches at finish line)
                mid_dist = tel["Distance"].max() * 0.5
                closest_idx = (tel["Distance"] - mid_dist).abs().argmin()
                x = float(tel["X"].iloc[closest_idx])
                y = float(tel["Y"].iloc[closest_idx])

                pos = lap_row.get("Position", None)
                compound = lap_row.get("Compound", "UNKNOWN")

                lap_data[lap_num][driver] = {
                    "x": x,
                    "y": y,
                    "position": int(pos) if pd.notna(pos) else 99,
                    "compound": str(compound) if pd.notna(compound) else "UNKNOWN",
                }
            except Exception as e:
                continue

    return lap_data
    """
    Load telemetry and return car (x, y) positions for each lap.

    Returns:
        {
          lap_number: {
            'VER': {'x': 1234, 'y': 5678, 'position': 1, 'compound': 'SOFT'},
            'NOR': {...},
            ...
          }
        }
    """
    fastf1.Cache.enable_cache("f1_cache")

    session = fastf1.get_session(year, round_num, session_type)
    session.load(telemetry=True, weather=False, messages=False)

    laps = session.laps
    max_lap = int(laps["LapNumber"].max())

    lap_data = {}

    for lap_num in range(1, max_lap + 1):
        lap_data[lap_num] = {}
        lap_slice = laps[laps["LapNumber"] == lap_num]

        for _, lap_row in lap_slice.iterrows():
            driver = lap_row["Driver"]
            try:
                tel = lap_row.get_telemetry()
                if tel is None or tel.empty:
                    continue

                # Take midpoint of telemetry as representative position
                mid = len(tel) // 2
                x = float(tel["X"].iloc[mid])
                y = float(tel["Y"].iloc[mid])

                # Get finishing position for this lap
                pos = lap_row.get("Position", None)
                compound = lap_row.get("Compound", "UNKNOWN")

                lap_data[lap_num][driver] = {
                    "x": x,
                    "y": y,
                    "position": int(pos) if pd.notna(pos) else 99,
                    "compound": compound if pd.notna(compound) else "UNKNOWN",
                }
            except Exception:
                continue

    return lap_data


def get_track_outline(year: int, round_num: int, session_type: str = "R"):
    """
    Returns (x_coords, y_coords) arrays tracing the full circuit outline.
    Uses the fastest lap's telemetry as the reference line.
    """
    fastf1.Cache.enable_cache("f1_cache")

    session = fastf1.get_session(year, round_num, session_type)
    session.load(telemetry=True, weather=False, messages=False)

    fastest = session.laps.pick_fastest()
    tel = fastest.get_telemetry()

    return tel["X"].values, tel["Y"].values
def get_driver_color(driver: str) -> tuple:
    """
    Returns an RGB tuple for each driver.
    Colors roughly match 2024 F1 team colors.
    """
    COLORS = {
        "VER": (30, 65, 255),    # Red Bull blue
        "PER": (30, 65, 255),
        "HAM": (0, 210, 190),    # Mercedes teal
        "RUS": (0, 210, 190),
        "LEC": (220, 0, 0),      # Ferrari red
        "SAI": (220, 0, 0),
        "NOR": (255, 135, 0),    # McLaren orange
        "PIA": (255, 135, 0),
        "ALO": (0, 110, 120),    # Aston green
        "STR": (0, 110, 120),
        "GAS": (0, 160, 222),    # Alpine blue
        "OCO": (0, 160, 222),
        "ALB": (0, 130, 250),    # Williams blue
        "SAR": (0, 130, 250),
        "TSU": (99, 0, 114),     # RB purple
        "LAW": (99, 0, 114),
        "HUL": (144, 0, 0),      # Haas red
        "MAG": (144, 0, 0),
        "BOT": (130, 0, 200),    # Sauber purple
        "ZHO": (130, 0, 200),
    }
    return COLORS.get(driver, (180, 180, 180))


def precompute_frames(lap_data: dict) -> list:
    """
    Flatten lap_data into a list of frames for smooth animation.
    Each frame = one lap snapshot.

    Returns:
        List of dicts: [{lap: int, drivers: {driver: {x,y,position,compound}}}, ...]
    """
    frames = []
    for lap_num in sorted(lap_data.keys()):
        frames.append({
            "lap": lap_num,
            "drivers": lap_data[lap_num],
        })
    return frames