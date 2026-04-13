"""
main.py
-------
Full pipeline entry point for the F1 Podium + Winner Predictor.

Usage:
    python main.py                  # full pipeline (collect → features → train → predict)
    python main.py --skip-collect   # skip data collection, use existing raw_data.csv
    python main.py --predict-only   # predict only, using saved models
"""

import argparse
from f1predictor.collect import setup_cache, collect_all
from f1predictor.features import build_features
from f1predictor.train import train
from f1predictor.predict import predict_race, print_predictions


# ── Edit these for the race you want to predict ───────────────────────────────
QUALIFYING = [
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
TRACK_TEMP = 30.0
IS_WET = False
# ─────────────────────────────────────────────────────────────────────────────


def main(skip_collect: bool = False, predict_only: bool = False):

    if not predict_only:
        if not skip_collect:
            print("\n" + "=" * 52)
            print("  STEP 1: DATA COLLECTION")
            print("=" * 52)
            setup_cache()
            collect_all(seasons=[2022, 2023, 2024, 2025])

        print("\n" + "=" * 52)
        print("  STEP 2: FEATURE ENGINEERING")
        print("=" * 52)
        build_features()

        print("\n" + "=" * 52)
        print("  STEP 3: TRAIN PODIUM + WINNER MODELS")
        print("=" * 52)
        train()

    print("\n" + "=" * 52)
    print("  STEP 4: PREDICTION")
    print("=" * 52)
    results = predict_race(QUALIFYING, track_temp=TRACK_TEMP, is_wet=IS_WET)
    print_predictions(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Podium + Winner Predictor")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip data collection, use existing raw_data.csv")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip all training, use saved models to predict")
    args = parser.parse_args()
    main(skip_collect=args.skip_collect, predict_only=args.predict_only)
