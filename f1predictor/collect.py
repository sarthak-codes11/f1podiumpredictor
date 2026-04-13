"""
collect.py
----------
Handles all data collection from FastF1 API.
Loops through seasons and saves raw race + qualifying + weather data.
"""

import os
import time
import fastf1
import pandas as pd


def setup_cache(cache_dir: str = "f1_cache"):
    """Create cache directory and enable FastF1 caching."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    print(f"✅ Cache enabled at: {cache_dir}")


def collect_season(year: int, sleep_between: float = 3.0) -> list[dict]:
    """
    Collect all race data for a single season.

    Args:
        year: F1 season year (e.g. 2023)
        sleep_between: seconds to wait between races (avoids rate limiting)

    Returns:
        List of dicts, one per driver per race
    """
    records = []
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        try:
            # Load race session
            race = fastf1.get_session(year, round_num, "R")
            race.load(telemetry=False, weather=True, messages=False)

            # Load qualifying session
            quali = fastf1.get_session(year, round_num, "Q")
            quali.load(telemetry=False, weather=False, messages=False)

            # Weather
            weather = race.weather_data
            avg_temp = weather["TrackTemp"].mean()
            avg_rainfall = weather["Rainfall"].mean()

            # Qualifying positions
            quali_results = quali.results[["Abbreviation", "Position"]].rename(
                columns={"Position": "quali_pos"}
            )

            # Race results
            race_results = race.results[["Abbreviation", "Position", "GridPosition"]]
            merged = race_results.merge(quali_results, on="Abbreviation", how="left")

            for _, row in merged.iterrows():
                records.append(
                    {
                        "year": year,
                        "round": round_num,
                        "circuit": event_name,
                        "driver": row["Abbreviation"],
                        "grid_pos": row["GridPosition"],
                        "quali_pos": row.get("quali_pos", None),
                        "finish_pos": row["Position"],
                        "track_temp": avg_temp,
                        "rainfall": avg_rainfall,
                    }
                )

            print(f"  ✅ R{round_num:02d} — {event_name}")
            time.sleep(sleep_between)

        except Exception as e:
            print(f"  ⚠️  Skipping R{round_num:02d} ({event_name}): {e}")
            if "500 calls/h" in str(e):
                print("  ⏳ Rate limit hit — waiting 60 seconds...")
                time.sleep(60)

    return records


def collect_all(seasons: list[int], output_path: str = "data/raw_data.csv"):
    """
    Collect data across multiple seasons and save to CSV.

    Args:
        seasons: list of years, e.g. [2022, 2023, 2024]
        output_path: where to save the raw CSV
    """
    all_records = []

    for year in seasons:
        print(f"\n📅 Collecting {year} season...")
        records = collect_season(year)
        all_records.extend(records)
        print(f"  → {len(records)} rows collected for {year}")

    df = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n🏁 Done! {len(df)} total rows saved to {output_path}")
    return df


if __name__ == "__main__":
    setup_cache()
    collect_all(seasons=[2022, 2023, 2024])
