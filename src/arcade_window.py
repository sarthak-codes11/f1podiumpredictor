"""
arcade_window.py
----------------
Main Arcade window for the F1 race replay simulation.
Fixed for Arcade 3.x — uses ShapeElementList for track rendering
and arcade.Text objects instead of draw_text.
"""

import arcade
import numpy as np
import math
from src.replay_data import get_driver_color, precompute_frames
from src.hud import draw_leaderboard, draw_speed_bar
from src.live_predict import predict_for_lap
from src.ai_driver import DQNAgent, AICarState, DT
from src.ai_track import load_circuit, get_surface, get_friction, cast_lidar_ray, get_state
from src.ai_driver import SHORT_OFFSETS, LONG_OFFSETS, SHORT_RANGE, LONG_RANGE

WINDOW_W = 1280
WINDOW_H = 720
WINDOW_TITLE = "F1 Race Replay"

BASE_FRAME_INTERVAL = 2.0
SPEED_STEPS = [0.25, 0.5, 1.0, 2.0, 4.0]
MODE_REPLAY   = "replay"
MODE_AI       = "ai"

def _normalize_track(track_x, track_y, margin=100, offset_x=220):
    """Scale and shift track to fit window, leaving left margin for HUD."""
    tx = np.array(track_x, dtype=float)
    ty = np.array(track_y, dtype=float)

    tx -= tx.min()
    ty -= ty.min()

    avail_w = WINDOW_W - offset_x - margin
    avail_h = WINDOW_H - margin * 2

    scale_x = avail_w / (tx.max() + 1e-9)
    scale_y = avail_h / (ty.max() + 1e-9)
    scale = min(scale_x, scale_y)

    tx = tx * scale + offset_x
    ty = ty * scale + margin

    return tx.tolist(), ty.tolist()


def _map_car_position(driver_x, driver_y,
                      raw_track_x, raw_track_y,
                      norm_track_x, norm_track_y) -> tuple:
    raw_arr = np.column_stack([raw_track_x, raw_track_y])
    diffs = raw_arr - np.array([driver_x, driver_y])
    dists = np.einsum("ij,ij->i", diffs, diffs)
    idx = int(np.argmin(dists))
    return float(norm_track_x[idx]), float(norm_track_y[idx])


class F1ReplayWindow(arcade.Window):

    def __init__(self, frames, raw_track_x, raw_track_y,
                 lap_data, track_temp=30.0, is_wet=False):
        super().__init__(WINDOW_W, WINDOW_H, WINDOW_TITLE, resizable=False)
        arcade.set_background_color((15, 15, 15))

        self.frames = frames
        self.raw_track_x = list(raw_track_x)
        self.raw_track_y = list(raw_track_y)
        self.lap_data = lap_data
        self.track_temp = track_temp
        self.is_wet = is_wet

        self.norm_track_x, self.norm_track_y = _normalize_track(
            raw_track_x, raw_track_y
        )

        self.current_frame = 0
        self.paused = False
        self.speed_idx = 2
        self.elapsed = 0.0
        self.predictions = {}

     

        # Text objects for HUD (faster than draw_text)
        self.lap_text = arcade.Text("", 600, 10, arcade.color.WHITE, 13, bold=True)
        self.speed_text = arcade.Text("", 10, 10, arcade.color.WHITE, 13, bold=True)
        self.legend_text = arcade.Text(
            "SPACE: pause   ←/→: lap   ↑/↓: speed   1-5: set speed   R: restart",
            780, 10, arcade.color.LIGHT_GRAY, 10
        )

        # ── AI Driver mode ────────────────────────────────────────────────────
        self.mode      = MODE_REPLAY
        self.ai_agent  = None
        self.ai_car    = None
        self.ai_cx     = None
        self.ai_cy     = None
        self.ai_track_w = 1.0
        self.ai_track_h = 1.0
        self.show_lidar = True
        self.ai_year   = None
        self.ai_round  = None

    def setup(self):
        """Build track geometry after OpenGL context is ready."""
        self.track_points = list(zip(self.norm_track_x, self.norm_track_y))
        self._update_predictions()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_speed(self):
        return SPEED_STEPS[self.speed_idx]

    @property
    def frame_interval(self):
        return BASE_FRAME_INTERVAL / self.current_speed

    @property
    def current_lap(self):
        return self.frames[self.current_frame]["lap"]

    @property
    def max_lap(self):
        return self.frames[-1]["lap"]

    @property
    def current_drivers(self):
        return self.frames[self.current_frame]["drivers"]

    # ── Prediction ────────────────────────────────────────────────────────────

    def _update_predictions(self):
        try:
            self.predictions = predict_for_lap(
                lap_positions=self.current_drivers,
                all_lap_history=self.lap_data,
                current_lap=self.current_lap,
                track_temp=self.track_temp,
                is_wet=self.is_wet,
            )
        except Exception:
            self.predictions = {}

    # ── Update loop ───────────────────────────────────────────────────────────

    def on_update(self, delta_time):
        if self.mode == MODE_AI:
            # AI mode runs at full 60fps — no lap-based stepping
            if not self.paused:
                self._update_ai()
            return

        # Replay mode
        if self.paused:
            return
        self.elapsed += delta_time
        if self.elapsed >= self.frame_interval:
            self.elapsed = 0.0
            if self.current_frame < len(self.frames) - 1:
                self.current_frame += 1
                self._update_predictions()

    # ── Draw ──────────────────────────────────────────────────────────────────

    def on_draw(self):
        self.clear()

        # Track
        self._draw_track()

        # Cars
        self._draw_cars()

        # HUD panels
        self._draw_hud_background()
        self._draw_leaderboard()
        self._draw_speed_bar()

    def _draw_track(self):
        """Draw circuit outline as individual line segments (Arcade 3.x compatible)."""
        pts = self.track_points
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            arcade.draw_line(x1, y1, x2, y2, (60, 60, 60), 6)
        # Close the loop
        arcade.draw_line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], (60, 60, 60), 6)

    def _draw_cars(self):
        for driver, data in self.current_drivers.items():
            try:
                rx, ry = data.get("x"), data.get("y")
                if rx is None or ry is None:
                    continue
                if rx == 0.0 and ry == 0.0:
                    continue

                sx, sy = _map_car_position(
                    rx, ry,
                    self.raw_track_x, self.raw_track_y,
                    self.norm_track_x, self.norm_track_y,
                )
            except Exception:
                continue

            r, g, b = get_driver_color(driver)

            arcade.draw_circle_outline(sx, sy, 10, (r, g, b), 2)
            arcade.draw_circle_filled(sx, sy, 7, (r, g, b))

            # Driver label using Text object
            arcade.Text(driver, sx + 11, sy - 4,
                        arcade.color.WHITE, 9).draw()

    def _draw_hud_background(self):
        arcade.draw_lrbt_rectangle_filled(0, 215, 35, WINDOW_H, (0, 0, 0, 190))
        arcade.draw_lrbt_rectangle_filled(0, WINDOW_W, 0, 34, (0, 0, 0, 210))

    def _draw_leaderboard(self):
        sorted_drivers = sorted(
            self.current_drivers.items(),
            key=lambda x: x[1].get("position", 99),
        )

        arcade.Text("LEADERBOARD", 10, WINDOW_H - 30,
                    arcade.color.WHITE, 11, bold=True).draw()
        arcade.Text("POS  DRV    WIN%   POD%", 10, WINDOW_H - 50,
                    arcade.color.LIGHT_GRAY, 9).draw()

        COMPOUND_COLORS = {
            "SOFT": arcade.color.RED,
            "MEDIUM": arcade.color.YELLOW,
            "HARD": arcade.color.WHITE,
            "INTERMEDIATE": arcade.color.GREEN,
            "WET": arcade.color.BLUE,
            "UNKNOWN": arcade.color.GRAY,
        }

        for i, (driver, data) in enumerate(sorted_drivers):
            y = WINDOW_H - 68 - (i * 21)
            if y < 40:
                break

            pos = data.get("position", 99)
            compound = data.get("compound", "UNKNOWN")
            pred = self.predictions.get(driver, {})
            win_p = pred.get("win_prob", 0.0)
            pod_p = pred.get("podium_prob", 0.0)

            if pos == 1:
                color = (255, 215, 0)
            elif pos == 2:
                color = (192, 192, 192)
            elif pos == 3:
                color = (205, 127, 50)
            else:
                color = arcade.color.WHITE

            arcade.Text(
                f"P{pos:<2} {driver:<4}",
                10, y, color, 10
            ).draw()

            dot_color = COMPOUND_COLORS.get(compound, arcade.color.GRAY)
            arcade.draw_circle_filled(140, y + 5, 5, dot_color)

            arcade.Text(
                f"{win_p:>5.1f}% {pod_p:>5.1f}%",
                148, y, arcade.color.LIGHT_GRAY, 9
            ).draw()

    def _draw_speed_bar(self):
        speed_labels = {0.25: "0.25x", 0.5: "0.5x", 1.0: "1x", 2.0: "2x", 4.0: "4x"}
        speed_label = speed_labels.get(self.current_speed, f"{self.current_speed}x")
        status = "⏸ PAUSED" if self.paused else f"▶ {speed_label}"

        arcade.Text(status, 10, 10, arcade.color.WHITE, 13, bold=True).draw()
        arcade.Text(
            f"LAP  {self.current_lap} / {self.max_lap}",
            WINDOW_W // 2 - 40, 10, arcade.color.WHITE, 13, bold=True
        ).draw()
        arcade.Text(
            "SPACE: pause   ←/→: lap   ↑/↓: speed   1-5: set speed   R: restart",
            760, 10, arcade.color.LIGHT_GRAY, 10
        ).draw()

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def _load_ai_mode(self, year: int, round_num: int):
        """Load AI agent + circuit centerline for AI driver mode."""
        if self.ai_agent is None:
            self.ai_agent = DQNAgent("models/best_time.pt")

        if self.ai_cx is None or self.ai_year != year or self.ai_round != round_num:
            self.ai_cx, self.ai_cy = load_circuit(year, round_num)
            self.ai_track_w = float(self.ai_cx.max() - self.ai_cx.min())
            self.ai_track_h = float(self.ai_cy.max() - self.ai_cy.min())
            self.ai_year  = year
            self.ai_round = round_num

        # Place car at start of circuit
        start_x = float(self.ai_cx[0])
        start_y = float(self.ai_cy[0])

        # Estimate starting angle from first two points
        if len(self.ai_cx) > 1:
            dx = self.ai_cx[1] - self.ai_cx[0]
            dy = self.ai_cy[1] - self.ai_cy[0]
            start_angle = math.atan2(dy, dx)
        else:
            start_angle = 0.0

        self.ai_car = AICarState(start_x, start_y, start_angle)
        print("✅ AI mode ready")

    def _update_ai(self):
        """Step the AI car forward one frame."""
        if self.ai_car is None or self.ai_agent is None:
            return

        state = get_state(
            self.ai_car.x, self.ai_car.y,
            self.ai_car.angle, self.ai_car.speed,
            self.ai_cx, self.ai_cy,
            self.ai_track_w, self.ai_track_h,
        )

        action = self.ai_agent.predict(state)
        accel, steer = self.ai_agent.action_to_inputs(action)

        surface = get_surface(self.ai_car.x, self.ai_car.y, self.ai_cx, self.ai_cy)
        friction = get_friction(surface)
        hit_wall = (surface == "wall")

        self.ai_car.step(accel, steer, friction, hit_wall)

    def _draw_ai_mode(self):
        """Draw AI car + LIDAR + HUD in AI driver mode."""
        import math

        # Draw track outline (reuse existing)
        self._draw_track()

        if self.ai_car is None:
            return

        # Map AI car world coords → screen coords
        sx, sy = _map_car_position(
            self.ai_car.x, self.ai_car.y,
            self.raw_track_x, self.raw_track_y,
            self.norm_track_x, self.norm_track_y,
        )

        # LIDAR rays
        if self.show_lidar and self.ai_cx is not None:
            for offset in SHORT_OFFSETS:
                d, hx, hy = cast_lidar_ray(
                    self.ai_car.x, self.ai_car.y,
                    self.ai_car.angle + offset,
                    SHORT_RANGE, self.ai_cx, self.ai_cy,
                )
                ex, ey = _map_car_position(
                    hx, hy,
                    self.raw_track_x, self.raw_track_y,
                    self.norm_track_x, self.norm_track_y,
                )
                arcade.draw_line(sx, sy, ex, ey, (255, 140, 0, 90), 1)
                arcade.draw_circle_filled(ex, ey, 3, (255, 140, 0))

            for offset in LONG_OFFSETS:
                d, hx, hy = cast_lidar_ray(
                    self.ai_car.x, self.ai_car.y,
                    self.ai_car.angle + offset,
                    LONG_RANGE, self.ai_cx, self.ai_cy,
                )
                ex, ey = _map_car_position(
                    hx, hy,
                    self.raw_track_x, self.raw_track_y,
                    self.norm_track_x, self.norm_track_y,
                )
                arcade.draw_line(sx, sy, ex, ey, (0, 120, 255, 60), 1)
                arcade.draw_circle_filled(ex, ey, 3, (0, 120, 255))

        # AI car dot
        arcade.draw_circle_outline(sx, sy, 12, (0, 255, 100), 3)
        arcade.draw_circle_filled(sx, sy, 8, (0, 255, 100))
        arcade.Text("AI", sx + 12, sy - 4, arcade.color.WHITE, 10).draw()

        # HUD
        self._draw_hud_background()
        arcade.draw_lrbt_rectangle_filled(0, 215, 35, WINDOW_H, (0, 0, 0, 190))

        arcade.Text("AI DRIVER MODE", 10, WINDOW_H - 30,
                    (0, 255, 100), 11, bold=True).draw()
        arcade.Text(f"Speed:  {abs(self.ai_car.speed):>6.1f}", 10, WINDOW_H - 55,
                    arcade.color.WHITE, 10).draw()
        arcade.Text(f"Lap Time: {self.ai_car.lap_time:>5.1f}s", 10, WINDOW_H - 75,
                    arcade.color.WHITE, 10).draw()
        arcade.Text(f"Wall Hits: {self.ai_car.wall_hits}", 10, WINDOW_H - 95,
                    arcade.color.ORANGE, 10).draw()

        best = f"{self.ai_car.best_lap:.1f}s" if self.ai_car.best_lap < 9999 else "---"
        arcade.Text(f"Best Lap: {best}", 10, WINDOW_H - 115,
                    arcade.color.YELLOW, 10).draw()

        lidar_status = "ON" if self.show_lidar else "OFF"
        arcade.Text(f"LIDAR: {lidar_status}  (L to toggle)", 10, WINDOW_H - 140,
                    arcade.color.LIGHT_GRAY, 9).draw()

        # Bottom bar
        arcade.draw_lrbt_rectangle_filled(0, WINDOW_W, 0, 34, (0, 0, 0, 210))
        arcade.Text("TAB: switch mode   SPACE: restart AI   L: toggle LIDAR   ESC: quit",
                    10, 10, arcade.color.LIGHT_GRAY, 10).draw()
        arcade.Text("▶ AI DRIVING",
                    WINDOW_W - 160, 10, (0, 255, 100), 13, bold=True).draw()

    def on_key_press(self, key, modifiers):

        # ── Mode switch ───────────────────────────────────────────────────────
        if key == arcade.key.TAB:
            if self.mode == MODE_REPLAY:
                self.mode = MODE_AI
                if hasattr(self, "_year") and hasattr(self, "_round"):
                    self._load_ai_mode(self._year, self._round)
            else:
                self.mode = MODE_REPLAY
            return

        # ── AI mode only ──────────────────────────────────────────────────────
        if self.mode == MODE_AI:
            if key == arcade.key.L:
                self.show_lidar = not self.show_lidar
            elif key == arcade.key.SPACE:
                if self.ai_car and self.ai_cx is not None:
                    import math
                    dx = self.ai_cx[1] - self.ai_cx[0]
                    dy = self.ai_cy[1] - self.ai_cy[0]
                    start_angle = math.atan2(dy, dx)
                    self.ai_car.reset(
                        float(self.ai_cx[0]),
                        float(self.ai_cy[0]),
                        start_angle,
                    )
            return

        # ── Replay mode only ──────────────────────────────────────────────────
        if key == arcade.key.SPACE:
            self.paused = not self.paused
        elif key == arcade.key.RIGHT:
            if self.current_frame < len(self.frames) - 1:
                self.current_frame += 1
                self._update_predictions()
                self.elapsed = 0.0
        elif key == arcade.key.LEFT:
            if self.current_frame > 0:
                self.current_frame -= 1
                self._update_predictions()
                self.elapsed = 0.0
        elif key == arcade.key.UP:
            self.speed_idx = min(self.speed_idx + 1, len(SPEED_STEPS) - 1)
        elif key == arcade.key.DOWN:
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif key == arcade.key.R:
            self.current_frame = 0
            self.elapsed = 0.0
            self._update_predictions()
        elif key == arcade.key.KEY_1:
            self.speed_idx = 0
        elif key == arcade.key.KEY_2:
            self.speed_idx = 1
        elif key == arcade.key.KEY_3:
            self.speed_idx = 2
        elif key == arcade.key.KEY_4:
            self.speed_idx = 3
        elif key == arcade.key.KEY_5:
            self.speed_idx = 4