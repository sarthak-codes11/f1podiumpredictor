"""
track_map.py
------------
Builds Plotly figures for:
  - Pre-race view: cars placed at grid positions on track outline
  - Replay view: cars at actual lap positions with win% color coding
"""

import numpy as np
import plotly.graph_objects as go


# Tyre compound colors matching F1 visuals
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
    "UNKNOWN": "#888888",
}


def _base_figure(track_x, track_y) -> go.Figure:
    """Create a dark-themed figure with the circuit outline drawn."""
    fig = go.Figure()

    # Circuit outline
    fig.add_trace(go.Scatter(
        x=track_x,
        y=track_y,
        mode="lines",
        line=dict(color="#333333", width=8),
        name="Track",
        hoverinfo="skip",
    ))

    fig.update_layout(
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        height=500,
    )
    return fig


def build_prerace_map(
    track_x,
    track_y,
    predictions_df,         # DataFrame with columns: driver, grid_pos, win_prob, podium_prob
) -> go.Figure:
    """
    Place cars at evenly spaced points along the start/finish straight,
    colored by win probability.
    """
    fig = _base_figure(track_x, track_y)

    # Space drivers along the first 5% of the track (start/finish area)
    n = len(predictions_df)
    start_idx = 0
    end_idx = max(1, int(len(track_x) * 0.05))
    indices = np.linspace(start_idx, end_idx, n, dtype=int)

    for i, (_, row) in enumerate(predictions_df.iterrows()):
        idx = indices[i]
        win_pct = row["win_prob"]

        # Color intensity based on win probability (green = high, grey = low)
        intensity = int((win_pct / 100) * 200) + 55
        color = f"rgb({255 - intensity}, {intensity}, 80)"

        fig.add_trace(go.Scatter(
            x=[track_x[idx]],
            y=[track_y[idx]],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(color="white", width=1)),
            text=[row["driver"]],
            textposition="top center",
            textfont=dict(color="white", size=10),
            name=row["driver"],
            hovertemplate=(
                f"<b>{row['driver']}</b><br>"
                f"Grid: P{int(row['grid_pos'])}<br>"
                f"Win: {row['win_prob']}%<br>"
                f"Podium: {row['podium_prob']}%"
                "<extra></extra>"
            ),
        ))

    return fig


def build_replay_map(
    track_x,
    track_y,
    lap_positions: dict,    # {driver: {x, y, position, compound}}
    lap_predictions: dict,  # {driver: {win_prob, podium_prob}}
    lap_num: int,
) -> go.Figure:
    """
    Draw car positions for a specific lap, colored by tyre compound.
    Win probability shown on hover.
    """
    fig = _base_figure(track_x, track_y)

    for driver, pos_data in lap_positions.items():
        compound = pos_data.get("compound", "UNKNOWN")
        color = COMPOUND_COLORS.get(compound, "#888888")
        pred = lap_predictions.get(driver, {})
        win_prob = pred.get("win_prob", 0.0)
        podium_prob = pred.get("podium_prob", 0.0)
        position = pos_data.get("position", 99)

        fig.add_trace(go.Scatter(
            x=[pos_data["x"]],
            y=[pos_data["y"]],
            mode="markers+text",
            marker=dict(
                size=14,
                color=color,
                line=dict(color="white", width=1),
            ),
            text=[driver],
            textposition="top center",
            textfont=dict(color="white", size=9),
            name=driver,
            hovertemplate=(
                f"<b>{driver}</b><br>"
                f"Position: P{position}<br>"
                f"Tyre: {compound}<br>"
                f"Win: {win_prob}%<br>"
                f"Podium: {podium_prob}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(title=dict(
        text=f"Lap {lap_num}",
        font=dict(color="white", size=14),
        x=0.02,
    ))

    return fig