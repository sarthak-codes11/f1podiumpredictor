"""
ai_track.py
-----------
Builds a driveable F1 circuit surface from FastF1 telemetry.
Replaces pixel-color wall detection (C++ version) with
distance-to-track-edge detection using the circuit polyline.

Track width is estimated from FastF1 data and used to determine
if the AI car is on track, on the edge, or out of bounds (wall).
"""

import math
import numpy as np
import fastf1


# Estimated half-width of F1 circuits in telemetry units
# FastF1 uses meters, typical F1 track is 12-15m wide
TRACK_HALF_WIDTH  = 6.0     # meters — on track
GRASS_HALF_WIDTH  = 10.0    # meters — on grass/runoff
WALL_HALF_WIDTH   = 13.0    # meters — wall/barrier


def load_circuit(year: int, round_num: int,
                 cache_dir: str = "f1_cache") -> tuple:
    """
    Load FastF1 session and extract circuit centerline polyline.

    Returns:
        (cx, cy) — numpy arrays of circuit centerline x/y in meters
    """
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, round_num, "R")
    session.load(telemetry=True, weather=False, messages=False)

    fastest = session.laps.pick_fastest()
    tel = fastest.get_telemetry()[["X", "Y"]].dropna()

    cx = tel["X"].values.astype(float)
    cy = tel["Y"].values.astype(float)

    print(f"✅ Circuit loaded: {len(cx)} centerline points")
    return cx, cy


def _dist_to_segment(px, py, ax, ay, bx, by) -> float:
    """Minimum distance from point P to segment AB."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx*abx + aby*aby
    if ab2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx*abx + apy*aby) / ab2))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


def dist_to_centerline(px: float, py: float,
                        cx: np.ndarray, cy: np.ndarray) -> float:
    """
    Fast nearest-distance from (px, py) to the circuit centerline polyline.
    Uses only nearby segments for speed.
    """
    # Find nearest point index first
    diffs = np.hypot(cx - px, cy - py)
    near_idx = int(np.argmin(diffs))

    # Check ±20 segments around nearest point
    best = float("inf")
    n = len(cx)
    for i in range(near_idx - 20, near_idx + 20):
        i = i % n
        j = (i + 1) % n
        d = _dist_to_segment(px, py, cx[i], cy[i], cx[j], cy[j])
        if d < best:
            best = d
    return best


def get_surface(px: float, py: float,
                cx: np.ndarray, cy: np.ndarray) -> str:
    """
    Returns surface type at position (px, py):
      'track'  — on track
      'grass'  — runoff / grass
      'wall'   — barrier hit
    """
    d = dist_to_centerline(px, py, cx, cy)
    if d <= TRACK_HALF_WIDTH:
        return "track"
    elif d <= GRASS_HALF_WIDTH:
        return "grass"
    elif d <= WALL_HALF_WIDTH:
        return "wall"
    else:
        return "wall"


def get_friction(surface: str) -> float:
    """Returns friction multiplier matching C++ GetFrictionMultiplier."""
    return {"track": 1.0, "grass": 3.0, "wall": 999.0}.get(surface, 1.0)


def cast_lidar_ray(px: float, py: float, angle: float,
                   max_dist: float,
                   cx: np.ndarray, cy: np.ndarray,
                   step: float = 3.0) -> tuple[float, float, float]:
    """
    Cast a LIDAR ray from (px, py) at given angle.
    Returns (distance, hit_x, hit_y) where distance <= max_dist.
    Stops when it hits a wall (dist_to_centerline > WALL_HALF_WIDTH).
    """
    dist = 0.0
    while dist < max_dist:
        rx = px + math.cos(angle) * dist
        ry = py + math.sin(angle) * dist
        if get_surface(rx, ry, cx, cy) == "wall":
            return dist, rx, ry
        dist += step
    rx = px + math.cos(angle) * max_dist
    ry = py + math.sin(angle) * max_dist
    return max_dist, rx, ry


def get_state(px: float, py: float, angle: float, speed: float,
              cx: np.ndarray, cy: np.ndarray,
              track_w: float, track_h: float) -> list[float]:
    """
    Build the 23-dimensional state vector matching C++ GetState().

    Args:
        px, py:    car position
        angle:     car heading (radians)
        speed:     current speed
        cx, cy:    circuit centerline arrays
        track_w/h: bounding box of circuit for position normalization
    """
    from src.ai_driver import (SHORT_OFFSETS, LONG_OFFSETS,
                                SHORT_RANGE, LONG_RANGE,
                                REFERENCE_DIST, MAX_SPEED)
    state = []

    # 1. Normalized speed
    state.append(speed / MAX_SPEED)

    # 2-3. Angle as sin/cos
    state.append(math.sin(angle))
    state.append(math.cos(angle))

    # 4-5. Normalized position
    state.append(px / track_w if track_w > 0 else 0.0)
    state.append(py / track_h if track_h > 0 else 0.0)

    # 6-18. Short-range LIDAR (danger, inverse normalized)
    for offset in SHORT_OFFSETS:
        d, _, _ = cast_lidar_ray(px, py, angle + offset, SHORT_RANGE, cx, cy)
        danger = 1.0 / ((d / REFERENCE_DIST) + 0.1)
        state.append(min(1.0, danger))

    # 19-23. Long-range LIDAR (distance normalized)
    for offset in LONG_OFFSETS:
        d, _, _ = cast_lidar_ray(px, py, angle + offset, LONG_RANGE, cx, cy)
        state.append(max(0.0, min(1.0, d / LONG_RANGE)))

    return state  # 23 dims