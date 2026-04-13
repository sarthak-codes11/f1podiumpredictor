"""
f1predictor
-----------
A machine learning package for predicting F1 podium finishers.

Modules:
    collect  — download race/quali/weather data from FastF1
    features — engineer ML features from raw data
    train    — train and evaluate the Random Forest model
    predict  — predict podium probabilities for upcoming races
"""

from f1predictor.collect import setup_cache, collect_all
from f1predictor.features import build_features
from f1predictor.train import train, load_models
from f1predictor.predict import predict_race, print_predictions

__all__ = [
    "setup_cache",
    "collect_all",
    "build_features",
    "train",
    "load_models",
    "predict_race",
    "print_predictions",
]