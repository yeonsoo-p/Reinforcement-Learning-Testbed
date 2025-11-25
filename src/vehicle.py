"""Car-like vehicle dynamics using bicycle model."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .config import VehicleConfig, VEHICLE_CONFIG


@dataclass
class VehicleState:
    """State of the vehicle."""
    x: float = 0.0          # x position
    y: float = 0.0          # y position
    theta: float = 0.0      # heading angle (radians)
    velocity: float = 0.0   # forward velocity
    steering: float = 0.0   # steering angle (radians)


class VehicleDynamics:
    """
    Bicycle model for car-like vehicle dynamics.

    The bicycle model simplifies the vehicle to two wheels:
    - Front wheel (steered)
    - Rear wheel (fixed)

    State: [x, y, theta, velocity]
    Control: [acceleration, steering_rate]
    """

    def __init__(
        self,
        config: Optional[VehicleConfig] = None,
        wheelbase: Optional[float] = None,
        max_steering: Optional[float] = None,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        max_steering_rate: Optional[float] = None,
        dt: Optional[float] = None,
    ):
        cfg = config or VEHICLE_CONFIG
        self.wheelbase = wheelbase if wheelbase is not None else cfg.wheelbase
        self.max_steering = max_steering if max_steering is not None else cfg.max_steering
        self.max_velocity = max_velocity if max_velocity is not None else cfg.max_velocity
        self.max_acceleration = max_acceleration if max_acceleration is not None else cfg.max_acceleration
        self.max_steering_rate = max_steering_rate if max_steering_rate is not None else cfg.max_steering_rate
        self.dt = dt if dt is not None else cfg.dt

        self.state = VehicleState()

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0, velocity: float = 0.0) -> np.ndarray:
        """Reset vehicle to initial state."""
        self.state = VehicleState(x=x, y=y, theta=theta, velocity=velocity, steering=0.0)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get current state as numpy array."""
        return np.array([
            self.state.x,
            self.state.y,
            self.state.theta,
            self.state.velocity,
            self.state.steering,
        ])

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Take a step with the given action.

        Args:
            action: [acceleration, steering_rate] normalized to [-1, 1]

        Returns:
            New state as numpy array
        """
        # Denormalize actions
        acceleration = action[0] * self.max_acceleration
        steering_rate = action[1] * self.max_steering_rate

        # Update steering angle
        new_steering = self.state.steering + steering_rate * self.dt
        new_steering = np.clip(new_steering, -self.max_steering, self.max_steering)

        # Update velocity
        new_velocity = self.state.velocity + acceleration * self.dt
        new_velocity = np.clip(new_velocity, 0.0, self.max_velocity)

        # Bicycle model kinematics
        # Using rear-axle reference point
        if abs(new_steering) > 1e-6:
            # With steering
            beta = np.arctan(0.5 * np.tan(new_steering))  # Slip angle at center

            # Update position and heading
            new_x = self.state.x + new_velocity * np.cos(self.state.theta + beta) * self.dt
            new_y = self.state.y + new_velocity * np.sin(self.state.theta + beta) * self.dt
            new_theta = self.state.theta + (new_velocity / self.wheelbase) * np.tan(new_steering) * self.dt
        else:
            # Straight line motion
            new_x = self.state.x + new_velocity * np.cos(self.state.theta) * self.dt
            new_y = self.state.y + new_velocity * np.sin(self.state.theta) * self.dt
            new_theta = self.state.theta

        # Normalize theta to [-pi, pi]
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

        # Update state
        self.state = VehicleState(
            x=new_x,
            y=new_y,
            theta=new_theta,
            velocity=new_velocity,
            steering=new_steering,
        )

        return self.get_state()

    def get_front_position(self) -> tuple[float, float]:
        """Get position of front axle."""
        front_x = self.state.x + self.wheelbase * np.cos(self.state.theta)
        front_y = self.state.y + self.wheelbase * np.sin(self.state.theta)
        return front_x, front_y

    def get_corners(self, width: float = 1.8) -> np.ndarray:
        """Get four corners of the vehicle for visualization."""
        length = self.wheelbase * 1.2  # Total length slightly more than wheelbase
        half_width = width / 2

        # Local corner positions (rear-left, rear-right, front-right, front-left)
        local_corners = np.array([
            [-length * 0.1, -half_width],
            [-length * 0.1, half_width],
            [length * 0.9, half_width],
            [length * 0.9, -half_width],
        ])

        # Rotation matrix
        cos_t = np.cos(self.state.theta)
        sin_t = np.sin(self.state.theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Transform to world coordinates
        world_corners = (rotation @ local_corners.T).T + np.array([self.state.x, self.state.y])

        return world_corners
