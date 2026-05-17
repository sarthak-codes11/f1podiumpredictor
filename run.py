"""
run.py
------
Root-level entry point. Run this instead of individual modules
to avoid Python package/module path conflicts.

Usage:
    python run.py features
    python run.py train
    python run.py predict --year 2026 --round 4 --circuit Miami --temp 45 --manual
    python run.py collect --seasons 2026
"""

import sys
from f1predictor.features import build_features
from f1predictor.train import train
from f1predictor.collect import collect_all, setup_cache


def run_features():
    build_features()


def run_train():
    train()


def run_collect(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022,2023,2024,2025,2026])
    parser.add_argument("--full", action="store_true")
    parsed = parser.parse_args(args)
    setup_cache()
    collect_all(seasons=parsed.seasons, incremental=not parsed.full)


def run_predict(args):
    import argparse
    import fastf1
    from f1predictor.predict import predict_race, fetch_qualifying, print_predictions

    parser = argparse.ArgumentParser()
    parser.add_argument("--year",    type=int,   default=2026)
    parser.add_argument("--round",   type=int,   default=4)
    parser.add_argument("--temp",    type=float, default=45.0)
    parser.add_argument("--wet",     action="store_true")
    parser.add_argument("--circuit", type=str,   default="Miami")
    parser.add_argument("--manual",  action="store_true")
    parsed = parser.parse_args(args)

    if parsed.manual:
        qualifying = [
            {"driver": "ANT", "grid_pos": 1,  "quali_pos": 1,  "team": "Mercedes"},
            {"driver": "RUS", "grid_pos": 2,  "quali_pos": 2,  "team": "Mercedes"},
            {"driver": "LEC", "grid_pos": 3,  "quali_pos": 3,  "team": "Ferrari"},
            {"driver": "HAM", "grid_pos": 4,  "quali_pos": 4,  "team": "Ferrari"},
            {"driver": "NOR", "grid_pos": 5,  "quali_pos": 5,  "team": "McLaren"},
            {"driver": "PIA", "grid_pos": 6,  "quali_pos": 6,  "team": "McLaren"},
            {"driver": "VER", "grid_pos": 7,  "quali_pos": 7,  "team": "Red Bull Racing"},
            {"driver": "HAD", "grid_pos": 8,  "quali_pos": 8,  "team": "Red Bull Racing"},
            {"driver": "BEA", "grid_pos": 9,  "quali_pos": 9,  "team": "Haas F1 Team"},
            {"driver": "GAS", "grid_pos": 10, "quali_pos": 10, "team": "Alpine"},
        ]
    else:
        fastf1.Cache.enable_cache("f1_cache")
        qualifying = fetch_qualifying(parsed.year, parsed.round)

    results = predict_race(
        qualifying,
        track_temp=parsed.temp,
        is_wet=parsed.wet,
        circuit_name=parsed.circuit,
        year=parsed.year,
    )
    print_predictions(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [features|train|predict|collect] [options]")
        sys.exit(1)

    command  = sys.argv[1]
    the_rest = sys.argv[2:]

    if command == "features":
        run_features()
    elif command == "train":
        run_train()
    elif command == "collect":
        run_collect(the_rest)
    elif command == "predict":
        run_predict(the_rest)
    else:
        print(f"Unknown command: {command}")
        print("Available: features, train, predict, collect")
        sys.exit(1)