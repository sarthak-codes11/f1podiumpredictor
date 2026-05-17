"""
collect.py
----------
Handles all data collection from FastF1 API.
Loops through seasons and saves raw race + qualifying + weather data.
Supports incremental collection — skips already-collected rounds.
"""

import os
import time
import argparse
from datetime import date

import fastf1
import pandas as pd


SEASONS = [2022, 2023, 2024, 2025, 2026]


def setup_cache(cache_dir: str = "f1_cache"):
    """Create cache directory and enable FastF1 caching."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    print(f"✅ Cache enabled at: {cache_dir}")


def get_completed_rounds(year: int) -> list[int]:
    """Return round numbers for races that have already happened."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    today = date.today()
    completed = schedule[schedule["EventDate"].dt.date < today]
    return completed["RoundNumber"].tolist()


def get_already_collected(output_path: str) -> set[tuple[int, int]]:
    """Read existing CSV and return set of (year, round) already collected."""
    if not os.path.exists(output_path):
        return set()
    df = pd.read_csv(output_path)
    return set(zip(df["year"], df["round"]))


def collect_season(
    year: int,
    skip_rounds: set = None,
    sleep_between: float = 3.0,
) -> list[dict]:
    """Collect all completed race data for a single season."""
    if skip_rounds is None:
        skip_rounds = set()

    records = []
    completed_rounds = get_completed_rounds(year)

    if not completed_rounds:
        print(f"  ⚠️  No completed rounds found for {year} yet.")
        return records

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        round_num  = int(event["RoundNumber"])
        event_name = event["EventName"]

        if round_num not in completed_rounds:
            continue

        if (year, round_num) in skip_rounds:
            print(f"  ⏭️  R{round_num:02d} — {event_name} (already collected, skipping)")
            continue

        try:
            race = fastf1.get_session(year, round_num, "R")
            race.load(telemetry=False, weather=True, messages=False)

            quali = fastf1.get_session(year, round_num, "Q")
            quali.load(telemetry=False, weather=False, messages=False)

            weather      = race.weather_data
            avg_temp     = weather["TrackTemp"].mean() if not weather.empty else None
            avg_rainfall = weather["Rainfall"].mean()  if not weather.empty else 0.0
            is_wet       = int(avg_rainfall > 0.1)     if avg_rainfall is not None else 0

            quali_results = quali.results[["Abbreviation", "Position"]].rename(
                columns={"Position": "quali_pos"}
            )

            race_results = race.results[
                ["Abbreviation", "Position", "GridPosition", "TeamName"]
            ]
            merged = race_results.merge(quali_results, on="Abbreviation", how="left")

            for _, row in merged.iterrows():
                records.append({
                    "year":      year,
                    "round":     round_num,
                    "circuit":   event_name,
                    "driver":    row["Abbreviation"],
                    "team":      row.get("TeamName", "Unknown"),
                    "grid_pos":  row["GridPosition"],
                    "quali_pos": row.get("quali_pos", None),
                    "finish_pos": row["Position"],
                    "track_temp": avg_temp,
                    "rainfall":   avg_rainfall,
                    "is_wet":     is_wet,
                    "reg_era":    0 if year <= 2021 else (1 if year <= 2025 else 2),
                })

            print(f"  ✅ R{round_num:02d} — {event_name}")
            time.sleep(sleep_between)

        except Exception as e:
            print(f"  ⚠️  Skipping R{round_num:02d} ({event_name}): {e}")
            if "500 calls/h" in str(e):
                print("  ⏳ Rate limit hit — waiting 60 seconds...")
                time.sleep(60)

    return records


def collect_all(
    seasons: list[int] = None,
    output_path: str = "data/raw_data.csv",
    incremental: bool = True,
) -> pd.DataFrame:
    """
    Collect data across multiple seasons and save to CSV.
    In incremental mode, appends only new rounds to the existing CSV.
    """
    if seasons is None:
        seasons = SEASONS

    already_collected = get_already_collected(output_path) if incremental else set()
    if already_collected:
        print(f"📂 Incremental mode: {len(already_collected)} rounds already on disk, skipping those.")

    all_new_records = []

    for year in seasons:
        print(f"\n📅 Collecting {year} season...")
        records = collect_season(year, skip_rounds=already_collected)
        all_new_records.extend(records)
        print(f"  → {len(records)} new rows collected for {year}")

    if not all_new_records:
        print("\n✅ Nothing new to collect — you're up to date!")
        return pd.read_csv(output_path) if os.path.exists(output_path) else pd.DataFrame()

    new_df = pd.DataFrame(all_new_records)

    if incremental and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["year", "round", "driver"])
    else:
        combined_df = new_df

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"\n🏁 Done! {len(new_df)} new rows added → {len(combined_df)} total rows in {output_path}")
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect F1 race data via FastF1")
    parser.add_argument("--full",    action="store_true",
                        help="Re-collect everything from scratch")
    parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    parser.add_argument("--output",  type=str, default="data/raw_data.csv")
    args = parser.parse_args()

    setup_cache()
    collect_all(
        seasons=args.seasons,
        output_path=args.output,
        incremental=not args.full,
    )