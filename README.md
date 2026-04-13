# 🏎️ F1 Podium Predictor + Race Replay

A machine learning system that predicts Formula 1 podium finishers,
combined with a live race replay visualization and AI driver simulation.

## Features
- **Podium predictor** — Random Forest model predicting top 3 finishers
- **Winner model** — Separate model predicting race winner
- **Race replay** — Arcade window with real FastF1 telemetry
- **AI driver** — DQN agent driving a real F1 circuit (press TAB)
- **Streamlit UI** — Pre-race prediction interface

## Setup

### Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

### Collect data and train models
python main.py

### Launch Streamlit predictions
streamlit run app.py

### Launch race replay
python replay.py --year 2024 --round 12

## Controls (Replay Window)
- TAB — switch between F1 replay and AI driver mode
- SPACE — pause/resume (replay) or restart AI (AI mode)
- ← / → — step laps
- ↑ / ↓ — change speed
- L — toggle LIDAR rays (AI mode)
- R — restart replay

## Project Structure
f1podiumpredictor/
├── app.py               # Streamlit UI
├── main.py              # Pipeline runner
├── replay.py            # Arcade replay entry point
├── f1predictor/         # Core ML package
│   ├── collect.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── src/                 # Replay + AI modules
│   ├── arcade_window.py
│   ├── replay_data.py
│   ├── live_predict.py
│   ├── ai_driver.py
│   └── ai_track.py
└── models/
    └── best_time.pt     # Trained DQN model