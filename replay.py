"""
replay.py
---------
Entry point for the F1 Arcade race replay simulation.

Usage:
    python replay.py                        # prompts for year + round
    python replay.py --year 2024 --round 12
    python replay.py --year 2024 --round 12 --wet
"""

import argparse
import fastf1
import arcade
from src.replay_data import load_lap_positions, get_track_outline, precompute_frames
from src.arcade_window import F1ReplayWindow


def main(year: int, round_num: int, track_temp: float, is_wet: bool):
    print(f"\n🏎️  Loading {year} Round {round_num}...")

    fastf1.Cache.enable_cache("f1_cache")

    print("  → Loading lap positions (this may take a minute on first run)...")
    lap_data = load_lap_positions(year, round_num)

    print("  → Loading track outline...")
    raw_x, raw_y = get_track_outline(year, round_num)

    print("  → Building frames...")
    frames = precompute_frames(lap_data)

    print(f"  ✅ {len(frames)} laps loaded. Launching window...\n")
    print("Controls:")
    print("  SPACE       — pause / resume")
    print("  ← / →       — step back / forward one lap")
    print("  ↑ / ↓       — speed up / slow down")
    print("  1–5         — set speed directly (0.25x to 4x)")
    print("  R           — restart from lap 1\n")

    # Sanity check
    print("📍 Sample lap 1 positions:")
    first_frame = frames[0]
    for drv, d in list(first_frame["drivers"].items())[:3]:
        print(f"  {drv}: x={d['x']:.1f}, y={d['y']:.1f}, pos={d['position']}")

    # Create window ONCE, call setup(), then run
    window = F1ReplayWindow(
        frames=frames,
        raw_track_x=raw_x,
        raw_track_y=raw_y,
        lap_data=lap_data,
        track_temp=track_temp,
        is_wet=is_wet,
    )
    # Pass year/round so TAB can load AI mode for the same circuit
    window._year  = year
    window._round = round_num
    window.setup()
    arcade.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Race Replay")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--temp", type=float, default=30.0,
                        help="Track temperature for live predictions")
    parser.add_argument("--wet", action="store_true",
                        help="Mark race as wet for live predictions")
    args = parser.parse_args()

    year = args.year or int(input("Enter year (e.g. 2024): "))
    round_num = args.round or int(input("Enter round number (e.g. 12): "))

    main(year, round_num, args.temp, args.wet)