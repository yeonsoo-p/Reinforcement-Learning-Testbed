"""Curve Follower RL - A reinforcement learning agent that learns to follow curves."""

from .curves import (
    CurveGenerator,
    SineCurve,
    ArcCurve,
    SCurve,
    SpiralCurve,
    HairpinCurve,
    RandomCurve,
    create_curve,
)
from .vehicle import VehicleDynamics
from .env import CurveFollowingEnv
from .config import (
    VehicleConfig,
    EnvConfig,
    RewardConfig,
    CurveParams,
    TrainingConfig,
    NetworkConfig,
    RendererConfig,
    VEHICLE_CONFIG,
    ENV_CONFIG,
    REWARD_CONFIG,
    CURVE_PARAMS,
    TRAINING_CONFIG,
    NETWORK_CONFIG,
    RENDERER_CONFIG,
)

__all__ = [
    # Curves
    "CurveGenerator",
    "SineCurve",
    "ArcCurve",
    "SCurve",
    "SpiralCurve",
    "HairpinCurve",
    "RandomCurve",
    "create_curve",
    # Vehicle
    "VehicleDynamics",
    # Environment
    "CurveFollowingEnv",
    # Config classes
    "VehicleConfig",
    "EnvConfig",
    "RewardConfig",
    "CurveParams",
    "TrainingConfig",
    "NetworkConfig",
    "RendererConfig",
    # Config instances
    "VEHICLE_CONFIG",
    "ENV_CONFIG",
    "REWARD_CONFIG",
    "CURVE_PARAMS",
    "TRAINING_CONFIG",
    "NETWORK_CONFIG",
    "RENDERER_CONFIG",
]
