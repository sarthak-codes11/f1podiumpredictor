"""
ai_driver.py
------------
DQN agent ported from C++ to Python.
Loads the trained best_time.pt model and runs inference.
Physics constants match the C++ trainer exactly.
"""

import math
import torch
import numpy as np


# ── Physics constants (must match C++ trainer exactly) ────────────────────────
MAX_SPEED         = 300.0
ACCELERATION      = 150.0
FRICTION          = 50.0
TURN_SPEED_BASE   = 3.0
TURN_SPEED_FACTOR = 0.3
DT                = 1.0 / 60.0

# ── LIDAR config ──────────────────────────────────────────────────────────────
SHORT_RANGE     = 200.0
LONG_RANGE      = 900.0
REFERENCE_DIST  = 50.0

SHORT_OFFSETS = [
    -math.pi/2,       # -90
    -5*math.pi/12,    # -75
    -math.pi/3,       # -60
    -math.pi/4,       # -45
    -math.pi/6,       # -30
    -math.pi/12,      # -15
     0.0,             #   0
     math.pi/12,      # +15
     math.pi/6,       # +30
     math.pi/4,       # +45
     math.pi/3,       # +60
     5*math.pi/12,    # +75
     math.pi/2,       # +90
]

LONG_OFFSETS = [
    -math.pi/6,   # -30
    -math.pi/12,  # -15
     0.0,         #   0
     math.pi/12,  # +15
     math.pi/6,   # +30
]

STATE_SIZE  = 23   # 5 base + 13 short LIDAR + 5 long LIDAR
ACTION_SIZE = 7


class DQNAgent:
    """
    Wraps the TorchScript DQN model for inference.
    Matches the C++ DQN action mapping exactly.
    """

    def __init__(self, model_path: str = "models/best_time.pt"):
        try:
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            print(f"✅ AI model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load AI model: {e}")
            self.model = None

    def predict(self, state: list[float]) -> int:
        """Return the best action given a state vector."""
        if self.model is None:
            return 0
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(t)
            return int(q_values.argmax(dim=1).item())

    @staticmethod
    def action_to_inputs(action: int) -> tuple[float, float]:
        """
        Map action index to (acceleration, steering).
        Matches C++ switch statement exactly.
        """
        mapping = {
            0: ( 1.0,  0.0),   # forward
            1: (-0.4,  0.0),   # reverse
            2: ( 0.0, -1.0),   # left
            3: ( 0.0,  1.0),   # right
            4: ( 1.0, -1.0),   # forward + left
            5: ( 1.0,  1.0),   # forward + right
            6: ( 0.0,  0.0),   # nothing
        }
        return mapping.get(action, (0.0, 0.0))


class AICarState:
    """
    Holds the physics state of the AI car and steps it forward each frame.
    """

    def __init__(self, start_x: float, start_y: float, start_angle: float = 0.0):
        self.x        = start_x
        self.y        = start_y
        self.angle    = start_angle
        self.speed    = 0.0
        self.vx       = 0.0
        self.vy       = 0.0
        self.lap_time = 0.0
        self.best_lap = float("inf")
        self.lap      = 0
        self.wall_hits = 0

    def reset(self, start_x: float, start_y: float, start_angle: float = 0.0):
        self.__init__(start_x, start_y, start_angle)

    def step(self, accel: float, steer: float,
             surface_friction: float = 1.0, hit_wall: bool = False):
        """
        Advance physics by one frame (DT seconds).
        Returns new (x, y) before wall correction is applied externally.
        """
        # Apply acceleration
        self.speed += accel * ACCELERATION * DT

        # Friction
        friction = FRICTION if accel != 0.0 else FRICTION * surface_friction
        if self.speed > 0:
            self.speed -= friction * DT
            if self.speed < 0:
                self.speed = 0.0
        elif self.speed < 0:
            self.speed += friction * DT
            if self.speed > 0:
                self.speed = 0.0

        # Speed clamp
        max_spd = MAX_SPEED * (0.5 if surface_friction > 2.0 else 1.0)
        self.speed = max(-max_spd * 0.5, min(max_spd, self.speed))

        # Steering
        if abs(self.speed) > 1.0:
            speed_factor = 1.0 / (1.0 + abs(self.speed) / MAX_SPEED * TURN_SPEED_FACTOR)
            turn_rate = TURN_SPEED_BASE * speed_factor
            self.angle += steer * turn_rate * DT * (self.speed / abs(self.speed))

        # Velocity
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed

        new_x = self.x + self.vx * DT
        new_y = self.y + self.vy * DT

        if hit_wall:
            self.speed *= -0.3
            self.wall_hits += 1
        else:
            self.x = new_x
            self.y = new_y

        self.lap_time += DT
        return self.x, self.y