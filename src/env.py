"""Gymnasium environment for curve following with car-like vehicle."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any

from .vehicle import VehicleDynamics
from .curves import CurveGenerator, create_curve
from .config import EnvConfig, RewardConfig, RendererConfig, ENV_CONFIG, REWARD_CONFIG, RENDERER_CONFIG


class CurveFollowingEnv(gym.Env):
    """
    Gymnasium environment for training a car-like agent to follow curves.

    Observation Space:
        - Cross-track error (distance from curve, signed)
        - Heading error (difference between vehicle heading and curve tangent)
        - Velocity
        - Steering angle
        - Curvature at closest point
        - Lookahead points (optional, for better anticipation)

    Action Space:
        - Acceleration (normalized to [-1, 1])
        - Steering rate (normalized to [-1, 1])

    Reward:
        - Negative reward for cross-track error
        - Negative reward for heading error
        - Small positive reward for progress along curve
        - Bonus for maintaining target velocity
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        curve_type: str = "sine",
        curve_kwargs: Optional[dict] = None,
        env_config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        render_config: Optional[RendererConfig] = None,
        max_cross_track_error: Optional[float] = None,
        target_velocity: Optional[float] = None,
        max_episode_steps: Optional[int] = None,
        num_lookahead_points: Optional[int] = None,
        lookahead_distance: Optional[float] = None,
        render_mode: Optional[str] = None,
        randomize_start: bool = True,
    ):
        """
        Initialize the environment.

        Args:
            curve_type: Type of curve ('sine', 'arc', 'scurve', 'spiral', 'hairpin', 'random')
            curve_kwargs: Arguments to pass to curve generator
            env_config: Environment configuration (uses defaults if None)
            reward_config: Reward configuration (uses defaults if None)
            render_config: Renderer configuration (uses defaults if None)
            max_cross_track_error: Maximum allowed cross-track error before termination
            target_velocity: Target velocity for the vehicle
            max_episode_steps: Maximum steps per episode
            num_lookahead_points: Number of lookahead points in observation
            lookahead_distance: Distance between lookahead points
            render_mode: 'human' or 'rgb_array'
            randomize_start: Whether to randomize starting position (adds noise to start)
        """
        super().__init__()

        # Load configs
        self._env_cfg = env_config or ENV_CONFIG
        self._reward_cfg = reward_config or REWARD_CONFIG
        self._render_cfg = render_config or RENDERER_CONFIG

        self.curve_type = curve_type
        self.curve_kwargs = curve_kwargs or {}
        self.max_cross_track_error = max_cross_track_error if max_cross_track_error is not None else self._env_cfg.max_cross_track_error
        self.target_velocity = target_velocity if target_velocity is not None else self._env_cfg.target_velocity
        self.max_episode_steps = max_episode_steps if max_episode_steps is not None else self._env_cfg.max_episode_steps
        self.num_lookahead_points = num_lookahead_points if num_lookahead_points is not None else self._env_cfg.num_lookahead_points
        self.lookahead_distance = lookahead_distance if lookahead_distance is not None else self._env_cfg.lookahead_distance
        self.render_mode = render_mode
        self.randomize_start = randomize_start

        # Create curve and vehicle
        self.curve = create_curve(curve_type, **self.curve_kwargs)
        self.vehicle = VehicleDynamics()

        # Action space: [acceleration, steering_rate] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Observation space
        # Base observations: cross_track, heading_error, velocity, steering, curvature, progress_ratio
        # Plus lookahead: for each lookahead point, relative x, y, and heading
        obs_dim = 6 + self.num_lookahead_points * 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Internal state
        self._step_count = 0
        self._total_progress = 0.0
        self._last_arc_length = 0.0

        # Rendering
        self._renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset curve (important for random curves)
        if self.curve_type == "random" or (options and options.get("new_curve", False)):
            self.curve.reset()
            _ = self.curve.points  # Trigger regeneration

        # Always start at the beginning of the curve
        start_idx = 0
        start_point = self.curve.points[start_idx].copy()
        start_heading = np.arctan2(
            self.curve.tangents[start_idx, 1],
            self.curve.tangents[start_idx, 0]
        )

        # Add small random offset for robustness (if randomize_start is True)
        if self.randomize_start:
            noise = self._env_cfg.start_position_noise
            start_point = start_point + self.np_random.uniform(-noise, noise, size=2)
            start_heading += self.np_random.uniform(
                -self._env_cfg.start_heading_noise,
                self._env_cfg.start_heading_noise
            )

        # Reset vehicle
        self.vehicle.reset(
            x=start_point[0],
            y=start_point[1],
            theta=start_heading,
            velocity=self.target_velocity * self._env_cfg.start_velocity_ratio,
        )

        self._step_count = 0
        self._total_progress = 0.0
        self._last_arc_length = self.curve.arc_lengths[start_idx]

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        # Clip action just in case
        action = np.clip(action, -1.0, 1.0)

        # Step vehicle dynamics
        self.vehicle.step(action)
        self._step_count += 1

        # Get current state
        position = np.array([self.vehicle.state.x, self.vehicle.state.y])
        heading = self.vehicle.state.theta
        velocity = self.vehicle.state.velocity

        # Compute errors
        cross_track, heading_error = self.curve.get_cross_track_error(position, heading)
        idx, distance, _ = self.curve.find_closest_point(position)
        current_arc = self.curve.arc_lengths[idx]

        # Compute progress (handle curve wrapping)
        progress = current_arc - self._last_arc_length
        if progress < -self.curve.total_length / 2:
            progress += self.curve.total_length
        elif progress > self.curve.total_length / 2:
            progress -= self.curve.total_length
        self._total_progress += max(0, progress)  # Only count forward progress
        self._last_arc_length = current_arc

        # Check if reached the end of the curve
        progress_ratio = current_arc / self.curve.total_length
        reached_end = (
            progress_ratio > self._reward_cfg.completion_progress_ratio
            and self._total_progress > self.curve.total_length * self._reward_cfg.completion_total_progress_ratio
        )

        # Compute reward
        reward = self._compute_reward(cross_track, heading_error, velocity, progress, action)

        # Add completion bonus if reached end
        if reached_end:
            reward += self._reward_cfg.completion_bonus

        # Check termination conditions
        off_track = abs(cross_track) > self.max_cross_track_error
        terminated = off_track or reached_end
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = self._get_info()
        info["cross_track_error"] = cross_track
        info["heading_error"] = heading_error
        info["progress"] = self._total_progress
        info["progress_ratio"] = progress_ratio
        info["reached_end"] = reached_end
        info["off_track"] = off_track

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Compute observation vector."""
        position = np.array([self.vehicle.state.x, self.vehicle.state.y])
        heading = self.vehicle.state.theta

        # Base observations
        cross_track, heading_error = self.curve.get_cross_track_error(position, heading)
        idx, _, _ = self.curve.find_closest_point(position)
        curvature = self.curve.curvatures[idx]

        # Progress along curve (0 = start, 1 = end)
        progress_ratio = self.curve.arc_lengths[idx] / self.curve.total_length

        # Normalize observations
        obs = [
            cross_track / self.max_cross_track_error,  # Normalized cross-track error
            heading_error / np.pi,  # Normalized heading error
            self.vehicle.state.velocity / self.vehicle.max_velocity,  # Normalized velocity
            self.vehicle.state.steering / self.vehicle.max_steering,  # Normalized steering
            np.clip(curvature * 10, -1, 1),  # Scaled curvature
            progress_ratio,  # Progress along curve (0 to 1)
        ]

        # Lookahead observations (relative to vehicle frame)
        cos_h = np.cos(-heading)
        sin_h = np.sin(-heading)

        for i in range(1, self.num_lookahead_points + 1):
            lookahead_dist = i * self.lookahead_distance
            lookahead_point, lookahead_heading = self.curve.get_lookahead_point(position, lookahead_dist)

            # Transform to vehicle frame
            rel_pos = lookahead_point - position
            local_x = rel_pos[0] * cos_h - rel_pos[1] * sin_h
            local_y = rel_pos[0] * sin_h + rel_pos[1] * cos_h

            # Relative heading
            rel_heading = lookahead_heading - heading
            rel_heading = np.arctan2(np.sin(rel_heading), np.cos(rel_heading))

            # Normalize
            obs.extend([
                local_x / (self.lookahead_distance * self.num_lookahead_points),
                local_y / self.max_cross_track_error,
                rel_heading / np.pi,
            ])

        return np.array(obs, dtype=np.float32)

    def _compute_reward(
        self,
        cross_track: float,
        heading_error: float,
        velocity: float,
        progress: float,
        action: np.ndarray,
    ) -> float:
        """Compute reward based on current state."""
        cfg = self._reward_cfg

        # Cross-track error penalty (quadratic with extra penalty beyond threshold)
        normalized_cte = abs(cross_track) / self.max_cross_track_error
        cross_track_penalty = cfg.cte_penalty_base * (normalized_cte ** 2)
        if normalized_cte > cfg.cte_penalty_threshold:
            cross_track_penalty += cfg.cte_penalty_extra * (normalized_cte - cfg.cte_penalty_threshold) ** 2

        # Heading error penalty
        normalized_heading = abs(heading_error) / np.pi
        heading_penalty = cfg.heading_penalty * (normalized_heading ** 2)

        # Precision bonus: Reward for staying very close to the curve
        if normalized_cte < cfg.precision_threshold_high:
            precision_bonus = cfg.precision_bonus_high
        elif normalized_cte < cfg.precision_threshold_low:
            precision_bonus = cfg.precision_bonus_low
        else:
            precision_bonus = 0.0

        # Simple velocity reward: reward going fast, let agent learn when to slow down
        # The CTE penalty provides the counter-pressure to slow on curves
        normalized_velocity = velocity / self.vehicle.max_velocity
        raw_speed_bonus = cfg.raw_speed_bonus_weight * normalized_velocity

        # Velocity matching reward (reaching target velocity)
        velocity_error = abs(velocity - self.target_velocity) / self.target_velocity
        velocity_reward = cfg.velocity_reward_weight * (1.0 - min(velocity_error, 1.0))

        # Progress reward (encourage forward motion)
        progress_reward = cfg.progress_reward_weight * max(0, progress)

        # Stopping penalty: penalize very low velocities
        min_acceptable_velocity = self.target_velocity * cfg.min_acceptable_velocity_ratio
        if velocity < min_acceptable_velocity:
            stopping_penalty = cfg.stopping_penalty * (1.0 - velocity / min_acceptable_velocity) ** 2
        else:
            stopping_penalty = 0.0

        # Circling/backward penalty: penalize negative or zero progress
        if progress <= 0:
            no_progress_penalty = cfg.no_progress_penalty_zero if progress == 0 else cfg.no_progress_penalty_backward * abs(progress)
        else:
            no_progress_penalty = 0.0

        # Smoothness penalty (discourage jerky actions)
        action_penalty = cfg.action_penalty * np.sum(action ** 2)

        # Survival bonus
        survival_bonus = cfg.survival_bonus

        total_reward = (
            cross_track_penalty
            + heading_penalty
            + precision_bonus
            + velocity_reward
            + raw_speed_bonus
            + progress_reward
            + stopping_penalty
            + no_progress_penalty
            + action_penalty
            + survival_bonus
        )

        return float(total_reward)

    def _get_info(self) -> dict:
        """Get info dictionary."""
        return {
            "step": self._step_count,
            "vehicle_x": self.vehicle.state.x,
            "vehicle_y": self.vehicle.state.y,
            "vehicle_theta": self.vehicle.state.theta,
            "vehicle_velocity": self.vehicle.state.velocity,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from .renderer import OpenCVRenderer
            self._renderer = OpenCVRenderer(
                config=self._render_cfg,
                render_mode=self.render_mode,
            )

        return self._renderer.render(
            curve=self.curve,
            vehicle=self.vehicle,
            info=self._get_info(),
        )

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register the environment with Gymnasium
gym.register(
    id="CurveFollowing-v0",
    entry_point="src.env:CurveFollowingEnv",
    max_episode_steps=1000,
)
