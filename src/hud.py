"""
hud.py
------
Draws the HUD overlay on the Arcade window:
  - Leaderboard (left side)
  - Speed controls + lap counter (bottom)
  - Win/podium probabilities next to each driver
"""

import arcade

# HUD layout constants
HUD_X = 15
HUD_Y_START = 580
ROW_HEIGHT = 22
FONT = "Arial"

COMPOUND_COLORS = {
    "SOFT":         arcade.color.RED,
    "MEDIUM":       arcade.color.YELLOW,
    "HARD":         arcade.color.WHITE,
    "INTERMEDIATE": arcade.color.GREEN,
    "WET":          arcade.color.BLUE,
    "UNKNOWN":      arcade.color.GRAY,
}

SPEED_LABELS = {
    0.25: "0.25x",
    0.5:  "0.5x",
    1.0:  "1x",
    2.0:  "2x",
    4.0:  "4x",
}


def draw_leaderboard(sorted_drivers: list, predictions: dict):
    """
    Draw left-side leaderboard with driver, compound, win%, podium%.

    Args:
        sorted_drivers: list of (driver, data) tuples sorted by position
        predictions: {driver: {win_prob, podium_prob}}
    """
    # Background panel
    arcade.draw_lrbt_rectangle_filled(0, 230, 0, 620, (0, 0, 0, 180))

    arcade.draw_text("LEADERBOARD", HUD_X, HUD_Y_START + 10,
                     arcade.color.WHITE, 11, bold=True, font_name=FONT)

    headers = f"{'POS':<4} {'DRV':<5} {'WIN%':>6} {'POD%':>6}"
    arcade.draw_text(headers, HUD_X, HUD_Y_START - 12,
                     arcade.color.LIGHT_GRAY, 9, font_name=FONT)

    for i, (driver, data) in enumerate(sorted_drivers):
        y = HUD_Y_START - 28 - (i * ROW_HEIGHT)
        if y < 10:
            break

        pos = data.get("position", 99)
        compound = data.get("compound", "UNKNOWN")
        pred = predictions.get(driver, {})
        win_p = pred.get("win_prob", 0.0)
        pod_p = pred.get("podium_prob", 0.0)

        # Highlight top 3
        if pos == 1:
            row_color = (255, 215, 0)       # gold
        elif pos == 2:
            row_color = (192, 192, 192)     # silver
        elif pos == 3:
            row_color = (205, 127, 50)      # bronze
        else:
            row_color = arcade.color.WHITE

        compound_dot_color = COMPOUND_COLORS.get(compound, arcade.color.GRAY)

        # Position + driver name
        arcade.draw_text(
            f"P{pos:<2} {driver:<4}",
            HUD_X, y, row_color, 10, font_name=FONT
        )

        # Compound dot
        arcade.draw_circle_filled(145, y + 5, 5, compound_dot_color)

        # Win % and Podium %
        arcade.draw_text(
            f"{win_p:>5.1f}% {pod_p:>5.1f}%",
            152, y, arcade.color.LIGHT_GRAY, 9, font_name=FONT
        )


def draw_speed_bar(current_speed: float, current_lap: int,
                   max_lap: int, paused: bool, width: int):
    """
    Draw bottom HUD bar with speed setting, lap counter, and controls legend.
    """
    # Background bar
    arcade.draw_lrbt_rectangle_filled(
        0, width, 0, 40, (0, 0, 0, 200)
    )

    # Speed label
    speed_label = SPEED_LABELS.get(current_speed, f"{current_speed}x")
    status = "⏸ PAUSED" if paused else f"▶ {speed_label}"
    arcade.draw_text(status, 10, 12, arcade.color.WHITE, 13,
                     bold=True, font_name=FONT)

    # Lap counter
    arcade.draw_text(
        f"LAP  {current_lap} / {max_lap}",
        width // 2 - 40, 12,
        arcade.color.WHITE, 13, bold=True, font_name=FONT
    )

    # Controls legend
    legend = "SPACE: pause    ←/→: lap    ↑/↓: speed    R: restart"
    arcade.draw_text(legend, width - 500, 12,
                     arcade.color.LIGHT_GRAY, 10, font_name=FONT)