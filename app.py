"""
app.py
------
Unified Streamlit UI combining:
  - Pre-race podium predictions with track map
  - Lap-by-lap race replay with live prediction updates
"""

import streamlit as st
import pandas as pd
from f1predictor.predict import predict_race
from f1predictor.features import get_driver_encoding
from src.replay_data import load_lap_positions, get_track_outline
from src.track_map import build_prerace_map, build_replay_map
from src.live_predict import predict_for_lap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Predictor + Replay",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ F1 Predictor + Race Replay")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🗓️ Session")
    year = st.selectbox("Year", [2025, 2024, 2023, 2022], index=1)
    round_num = st.number_input("Round Number", min_value=1, max_value=24, value=12)

    st.divider()
    st.header("🌤️ Conditions")
    track_temp = st.slider("Track Temp (°C)", 10, 60, 30)
    is_wet = st.checkbox("Wet Race")

    st.divider()
    st.header("📋 Qualifying Order")
    st.caption("Enter top 10 for predictions")

    try:
        known_drivers = sorted(get_driver_encoding("data/features.csv").keys())
    except FileNotFoundError:
        known_drivers = [
            "VER", "NOR", "LEC", "SAI", "HAM",
            "RUS", "PIA", "ALO", "PER", "STR",
        ]

    qualifying = []
    for i in range(10):
        driver = st.selectbox(
            f"P{i+1:02d}",
            options=known_drivers,
            index=min(i, len(known_drivers) - 1),
            key=f"q_{i}",
        )
        qualifying.append({"driver": driver, "grid_pos": i + 1, "quali_pos": i + 1})

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Pre-Race Predictions", "▶️ Race Replay"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRE-RACE PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running models..."):
            try:
                predictions = predict_race(
                    qualifying_order=qualifying,
                    track_temp=track_temp,
                    is_wet=is_wet,
                )

                # Top 3 metric cards
                st.subheader("🏆 Predicted Podium")
                medals = ["🏆", "🥈", "🥉"]
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    row = predictions.iloc[i]
                    col.metric(
                        label=f"{medals[i]} {row['driver']}",
                        value=f"{row['win_prob']}% Win",
                        delta=f"{row['podium_prob']}% Podium",
                    )

                st.divider()

                # Track map
                st.subheader("🗺️ Grid Map — colored by Win Probability")
                with st.spinner("Loading track outline..."):
                    try:
                        tx, ty = get_track_outline(year, round_num)
                        fig = build_prerace_map(tx, ty, predictions)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Track map unavailable: {e}")

                st.divider()

                # Full rankings table
                st.subheader("📊 Full Rankings")
                for i, row in predictions.iterrows():
                    medal = medals[i] if i < 3 else f"  P{i+1}"
                    c1, c2, c3, c4 = st.columns([1, 2, 3, 3])
                    c1.markdown(f"**{medal}**")
                    c2.markdown(f"**{row['driver']}** · Grid P{int(row['grid_pos'])}")
                    c3.markdown(f"Win: **{row['win_prob']}%**")
                    c3.progress(int(row["win_prob"]))
                    c4.markdown(f"Podium: **{row['podium_prob']}%**")
                    c4.progress(int(row["podium_prob"]))

            except FileNotFoundError as e:
                st.error(f"Model not found: {e} — run `python main.py --skip-collect` first.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RACE REPLAY
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.caption("Loads real telemetry from FastF1. Takes a minute on first load.")

    if st.button("▶️ Load Race Data", type="primary", use_container_width=True):
        with st.spinner(f"Loading {year} Round {round_num} telemetry (this takes a while)..."):
            try:
                # Cache replay data in session state so it doesn't reload on every interaction
                cache_key = f"replay_{year}_{round_num}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = load_lap_positions(year, round_num)
                    tx, ty = get_track_outline(year, round_num)
                    st.session_state[f"track_{year}_{round_num}"] = (tx, ty)

                st.success("✅ Telemetry loaded!")

            except Exception as e:
                st.error(f"Failed to load telemetry: {e}")

    # Show replay controls only if data is loaded
    cache_key = f"replay_{year}_{round_num}"
    if cache_key in st.session_state:
        lap_data = st.session_state[cache_key]
        tx, ty = st.session_state[f"track_{year}_{round_num}"]
        max_lap = max(lap_data.keys())

        st.subheader("🎮 Replay Controls")
        lap_num = st.slider("Lap", min_value=1, max_value=max_lap, value=1)

        col1, col2 = st.columns(2)
        col1.metric("Current Lap", f"{lap_num} / {max_lap}")

        current_lap_positions = lap_data.get(lap_num, {})

        # Live predictions for this lap
        with st.spinner("Updating predictions..."):
            try:
                live_preds = predict_for_lap(
                    lap_positions=current_lap_positions,
                    all_lap_history=lap_data,
                    current_lap=lap_num,
                    track_temp=track_temp,
                    is_wet=is_wet,
                )
            except Exception:
                live_preds = {}

        # Track map for this lap
        st.subheader(f"🗺️ Track Map — Lap {lap_num}")
        fig = build_replay_map(tx, ty, current_lap_positions, live_preds, lap_num)
        st.plotly_chart(fig, use_container_width=True)

        # Live leaderboard
        st.subheader("📋 Live Leaderboard + Predictions")
        sorted_drivers = sorted(
            current_lap_positions.items(),
            key=lambda x: x[1].get("position", 99),
        )

        medals = ["🏆", "🥈", "🥉"]
        for i, (driver, data) in enumerate(sorted_drivers):
            pred = live_preds.get(driver, {})
            medal = medals[i] if i < 3 else f"P{i+1:02d}"
            compound = data.get("compound", "?")
            win_p = pred.get("win_prob", 0.0)
            pod_p = pred.get("podium_prob", 0.0)

            c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 3, 3])
            c1.markdown(f"**{medal}**")
            c2.markdown(f"**{driver}**")
            c3.markdown(f"🔴 {compound}")
            c4.markdown(f"Win: **{win_p}%**")
            c4.progress(int(win_p))
            c5.markdown(f"Podium: **{pod_p}%**")
            c5.progress(int(pod_p))