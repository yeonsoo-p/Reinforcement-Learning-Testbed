"""Configuration constants for the curve following RL environment."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


# =============================================================================
# Vehicle Dynamics Configuration
# =============================================================================
@dataclass
class VehicleConfig:
    """Configuration for vehicle dynamics."""
    wheelbase: float = 2.5              # Distance between front and rear axles (m)
    max_steering: float = np.pi / 4     # Maximum steering angle (45 degrees)
    max_velocity: float = 20.0          # Maximum velocity (m/s)
    max_acceleration: float = 6.0       # Maximum acceleration (m/s²)
    max_steering_rate: float = np.pi / 4  # Maximum steering rate (45°/s)
    dt: float = 0.1                     # Simulation time step (s)


# =============================================================================
# Environment Configuration
# =============================================================================
@dataclass
class EnvConfig:
    """Configuration for the curve following environment."""
    max_cross_track_error: float = 2.0    # Maximum allowed CTE before termination (m) - strict!
    target_velocity: float = 12.0         # Target velocity (m/s) - high speed goal
    max_episode_steps: int = 1000         # Maximum steps per episode
    num_lookahead_points: int = 80        # Number of lookahead points
    lookahead_distance: float = 0.5       # Distance between lookahead points (m)

    # Starting position randomization
    start_position_noise: float = 0.5     # Random offset range for start position (m)
    start_heading_noise: float = 0.05     # Random offset range for start heading (rad)
    start_velocity_ratio: float = 0.5     # Initial velocity as ratio of target


# =============================================================================
# Reward Configuration
# =============================================================================
@dataclass
class RewardConfig:
    """Configuration for reward function weights and thresholds.

    Tuned for: MAXIMUM SPEED while STRICTLY staying on path.
    """
    # Cross-track error penalty - STRICT path following
    cte_penalty_base: float = -10.0         # Base quadratic penalty coefficient (was -5.0)
    cte_penalty_threshold: float = 0.15     # Threshold for extra penalty - tighter (was 0.3)
    cte_penalty_extra: float = -30.0        # Extra penalty beyond threshold (was -2.0)

    # Heading error penalty
    heading_penalty: float = -2.0           # Quadratic heading penalty coefficient (was -1.5)

    # Precision bonus thresholds (normalized CTE) - reduced, speed > precision within bounds
    precision_threshold_high: float = 0.1   # Threshold for high precision bonus
    precision_threshold_low: float = 0.2    # Threshold for low precision bonus
    precision_bonus_high: float = 0.1       # Bonus for high precision (was 0.5)
    precision_bonus_low: float = 0.0        # Bonus for low precision (was 0.2)

    # Velocity rewards - agent learns optimal speed through trial and error
    velocity_reward_weight: float = 1.0     # Reward for reaching target velocity
    raw_speed_bonus_weight: float = 0.3     # Direct bonus for going fast (normalized by max_velocity)

    # Progress reward - STRONG incentive for forward motion
    progress_reward_weight: float = 3.0     # Weight for forward progress (was 0.15)

    # Stopping penalty - only penalize near-zero velocity (actual stopping)
    min_acceptable_velocity_ratio: float = 0.1  # Only penalize very slow movement
    stopping_penalty: float = -2.0              # Penalty for nearly stopping

    # No progress penalty
    no_progress_penalty_zero: float = -1.0      # Penalty for zero progress (was -0.5)
    no_progress_penalty_backward: float = -2.0  # Penalty coefficient for backward (was -1.0)

    # Action smoothness penalty
    action_penalty: float = -0.02           # Penalty coefficient for action magnitude (was -0.03)

    # Survival bonus
    survival_bonus: float = 0.2             # Bonus for staying on track (was 0.3)

    # Completion bonus - big reward for finishing fast
    completion_bonus: float = 50.0          # Bonus for completing the curve (was 10.0)

    # Completion thresholds
    completion_progress_ratio: float = 0.95     # Progress ratio to consider complete
    completion_total_progress_ratio: float = 0.9  # Total progress ratio required


# =============================================================================
# Curve Parameters
# =============================================================================
@dataclass
class CurveParams:
    """Default parameters for each curve type."""
    sine: Dict[str, Any] = field(default_factory=lambda: {
        "amplitude": 20.0,
        "wavelength": 50.0,
        "num_periods": 3.0,
    })
    arc: Dict[str, Any] = field(default_factory=lambda: {
        "radius": 35.0,
        "arc_angle": np.pi * 1.5,  # 270 degrees
    })
    scurve: Dict[str, Any] = field(default_factory=lambda: {
        "length": 120.0,
        "amplitude": 35.0,
    })
    spiral: Dict[str, Any] = field(default_factory=lambda: {
        "initial_radius": 8.0,
        "growth_rate": 4.0,
        "num_turns": 2.0,
    })
    hairpin: Dict[str, Any] = field(default_factory=lambda: {
        "num_turns": 3,
        "turn_radius": 12.0,
        "straight_length": 50.0,
    })
    random: Dict[str, Any] = field(default_factory=lambda: {
        "num_control_points": 10,
        "bounds": (-60, 60, -60, 60),
    })

    def get(self, curve_type: str) -> Dict[str, Any]:
        """Get parameters for a curve type."""
        return getattr(self, curve_type, {})


# =============================================================================
# Training Configuration
# =============================================================================
@dataclass
class TrainingConfig:
    """Configuration for PPO training."""
    # PPO hyperparameters (defaults)
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Environment settings
    n_envs: int = 32

    # Checkpointing
    save_freq: int = 10_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 10

    # VecNormalize settings
    clip_obs: float = 10.0
    clip_reward: float = 10.0


# =============================================================================
# Neural Network Architecture Configuration
# =============================================================================
@dataclass
class NetworkConfig:
    """Configuration for the attention-based feature extractor."""
    features_dim: int = 128
    num_base_obs: int = 6
    num_lookahead_points: int = 80
    lookahead_features: int = 3
    embed_dim: int = 64
    num_heads: int = 8

    # Policy/Value network architecture
    pi_layers: tuple = (256, 256)
    vf_layers: tuple = (256, 256)


# =============================================================================
# Renderer Configuration
# =============================================================================
@dataclass
class RendererConfig:
    """Configuration for the OpenCV renderer."""
    width: int = 800
    height: int = 600

    # Colors (BGR format)
    background_color: tuple = (40, 40, 40)
    curve_color: tuple = (100, 100, 255)
    vehicle_color: tuple = (0, 255, 100)
    target_color: tuple = (255, 200, 0)
    trail_color: tuple = (100, 200, 100)

    # Camera settings
    pixels_per_meter: float = 8.0
    min_zoom: float = 1.0
    max_zoom: float = 50.0

    # Trail settings
    max_trail_length: int = 200

    # UI settings
    window_name: str = "Curve Following RL"
    render_fps: int = 30

    # Drawing margins
    visibility_margin: int = 100
    trail_margin: int = 50


# =============================================================================
# Default Configuration Instances
# =============================================================================
VEHICLE_CONFIG = VehicleConfig()
ENV_CONFIG = EnvConfig()
REWARD_CONFIG = RewardConfig()
CURVE_PARAMS = CurveParams()
TRAINING_CONFIG = TrainingConfig()
NETWORK_CONFIG = NetworkConfig()
RENDERER_CONFIG = RendererConfig()

# =============================================================================
# Configuration Validation
# =============================================================================
def _validate_configs():
    """Validate that related configs are consistent."""
    # Ensure network config matches environment config for observation dimensions
    if ENV_CONFIG.num_lookahead_points != NETWORK_CONFIG.num_lookahead_points:
        raise ValueError(
            f"Config mismatch: ENV_CONFIG.num_lookahead_points ({ENV_CONFIG.num_lookahead_points}) "
            f"!= NETWORK_CONFIG.num_lookahead_points ({NETWORK_CONFIG.num_lookahead_points}). "
            "These must match for the network to process observations correctly."
        )

_validate_configs()
