#!/usr/bin/env python3
"""Training script for the curve following RL agent."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.env import CurveFollowingEnv
from src.config import CURVE_PARAMS, TRAINING_CONFIG, NETWORK_CONFIG


class AttentionLookaheadExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that uses self-attention over lookahead points.

    Architecture:
        base_obs (num_base_obs) ─────────────────────────────────┐
                                                                 ├── Combined → features_dim
        lookahead (num_lookahead_points×3) → Self-Attention → pool ─┘

    The attention mechanism learns which lookahead points are most relevant
    for the current decision (e.g., immediate points for corrections,
    distant points for anticipating curves).

    Note: All parameters should be passed explicitly from NETWORK_CONFIG to ensure
    consistency with the environment's observation space.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int,
        num_base_obs: int,
        num_lookahead_points: int,
        lookahead_features: int,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__(observation_space, features_dim)

        self.num_base_obs = num_base_obs
        self.num_lookahead_points = num_lookahead_points
        self.lookahead_features = lookahead_features
        self.embed_dim = embed_dim

        # Base observations network
        self.base_net = nn.Sequential(
            nn.Linear(num_base_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Lookahead embedding: project each point (3 features) to embed_dim
        self.lookahead_embed = nn.Linear(lookahead_features, embed_dim)

        # Learnable positional encoding for sequence order
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_lookahead_points, embed_dim) * 0.1
        )

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Post-attention processing
        # For longer sequences, add an intermediate layer
        lookahead_flat_dim = embed_dim * num_lookahead_points
        self.lookahead_net = nn.Sequential(
            nn.Linear(lookahead_flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Combine base and lookahead features
        # 64 (base) + 64 (lookahead) = 128
        self.combine = nn.Sequential(
            nn.Linear(64 + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split observations into base and lookahead
        base_obs = observations[:, :self.num_base_obs]
        lookahead_flat = observations[:, self.num_base_obs:]

        # Reshape lookahead: (batch, num_lookahead_points * 3) -> (batch, num_lookahead_points, 3)
        lookahead = lookahead_flat.reshape(
            batch_size, self.num_lookahead_points, self.lookahead_features
        )

        # Process base observations
        base_features = self.base_net(base_obs)

        # Embed lookahead points: (batch, num_lookahead_points, 3) -> (batch, num_lookahead_points, embed_dim)
        lookahead_embedded = self.lookahead_embed(lookahead)

        # Add positional encoding
        lookahead_embedded = lookahead_embedded + self.pos_encoding

        # Self-attention over lookahead sequence
        # This lets the network learn relationships between points
        # (e.g., "point 3 curves sharply after point 2")
        attn_output, _ = self.attention(
            lookahead_embedded,
            lookahead_embedded,
            lookahead_embedded,
        )

        # Residual connection + layer norm
        lookahead_attended = self.layer_norm(lookahead_embedded + attn_output)

        # Flatten and process: (batch, num_lookahead_points, embed_dim) -> (batch, num_lookahead_points*embed_dim) -> (batch, 64)
        lookahead_flat = lookahead_attended.reshape(batch_size, -1)
        lookahead_features = self.lookahead_net(lookahead_flat)

        # Combine base and lookahead features
        combined = torch.cat([base_features, lookahead_features], dim=1)

        return self.combine(combined)


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_cross_track_errors = []

    def _on_step(self) -> bool:
        # Log episode statistics when episodes end
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

            if "cross_track_error" in info:
                self.episode_cross_track_errors.append(abs(info["cross_track_error"]))

        # Log mean metrics every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.episode_cross_track_errors) > 0:
            mean_cte = np.mean(self.episode_cross_track_errors[-100:])
            self.logger.record("custom/mean_cross_track_error", mean_cte)

        return True


def make_env(
    curve_type: str,
    curve_kwargs: dict,
    rank: int,
    seed: int = 0,
):
    """Create a single environment instance."""
    def _init():
        env = CurveFollowingEnv(
            curve_type=curve_type,
            curve_kwargs=curve_kwargs,
            render_mode=None,
            randomize_start=True,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    """Main training function."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.curve_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training configuration:")
    print(f"  Curve type: {args.curve_type}")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Number of environments: {args.n_envs}")
    print(f"  Output directory: {output_dir}")
    print()

    # Curve-specific parameters from config
    curve_kwargs = CURVE_PARAMS.get(args.curve_type)

    # Check if resuming from checkpoint
    resume_path = getattr(args, 'resume', None)

    # Create vectorized environments
    print("Creating training environments...")
    train_venv = make_vec_env(
        lambda: CurveFollowingEnv(
            curve_type=args.curve_type,
            curve_kwargs=curve_kwargs,
            render_mode=None,
            randomize_start=True,
        ),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_venv = make_vec_env(
        lambda: CurveFollowingEnv(
            curve_type=args.curve_type,
            curve_kwargs=curve_kwargs,
            render_mode=None,
            randomize_start=True,
        ),
        n_envs=1,
    )

    if resume_path:
        # Resume training from checkpoint
        resume_path = Path(resume_path)

        # Determine model and normalizer paths
        if resume_path.is_dir():
            model_file = resume_path / "final_model.zip"
            if not model_file.exists():
                model_file = resume_path / "best_model" / "best_model.zip"
            vec_normalize_path = resume_path / "vec_normalize.pkl"
        else:
            model_file = resume_path
            vec_normalize_path = resume_path.parent / "vec_normalize.pkl"

        print(f"Resuming training from: {model_file}")

        # Load VecNormalize statistics
        if vec_normalize_path.exists():
            print(f"Loading normalization stats from: {vec_normalize_path}")
            # Load for training env
            env = VecNormalize.load(str(vec_normalize_path), train_venv)
            env.training = True  # Enable updating statistics
            env.norm_reward = True

            # Also load for eval env (required for sync_envs_normalization)
            eval_env = VecNormalize.load(str(vec_normalize_path), eval_venv)
            eval_env.training = False  # Don't update during evaluation
            eval_env.norm_reward = False
        else:
            # No saved stats, create fresh VecNormalize
            env = VecNormalize(
                train_venv, norm_obs=True, norm_reward=True,
                clip_obs=TRAINING_CONFIG.clip_obs, clip_reward=TRAINING_CONFIG.clip_reward
            )
            eval_env = VecNormalize(
                eval_venv, norm_obs=True, norm_reward=False,
                clip_obs=TRAINING_CONFIG.clip_obs, training=False
            )

        # Load the model
        model = PPO.load(
            str(model_file),
            env=env,
            device="cpu",
            # Override hyperparameters if needed
            learning_rate=args.learning_rate,
            tensorboard_log=str(output_dir / "tensorboard"),
        )
        print(f"Model loaded successfully!")
    else:
        # Fresh training - wrap with VecNormalize
        env = VecNormalize(
            train_venv,
            norm_obs=True,
            norm_reward=True,
            clip_obs=TRAINING_CONFIG.clip_obs,
            clip_reward=TRAINING_CONFIG.clip_reward,
        )
        eval_env = VecNormalize(
            eval_venv,
            norm_obs=True,
            norm_reward=False,  # Don't normalize reward for evaluation
            clip_obs=TRAINING_CONFIG.clip_obs,
            training=False,  # Don't update statistics during evaluation
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_curve_follower",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=TRAINING_CONFIG.n_eval_episodes,
        deterministic=True,
    )

    tensorboard_callback = TensorboardCallback()

    if not resume_path:
        # Create new PPO model with attention-based feature extractor
        print("Creating PPO model with attention-based lookahead processing...")
        net_cfg = NETWORK_CONFIG
        policy_kwargs = {
            "features_extractor_class": AttentionLookaheadExtractor,
            "features_extractor_kwargs": {
                "features_dim": net_cfg.features_dim,
                "num_base_obs": net_cfg.num_base_obs,
                "num_lookahead_points": net_cfg.num_lookahead_points,
                "lookahead_features": net_cfg.lookahead_features,
                "embed_dim": net_cfg.embed_dim,
                "num_heads": net_cfg.num_heads,
            },
            # Policy and value networks after feature extraction
            "net_arch": dict(pi=list(net_cfg.pi_layers), vf=list(net_cfg.vf_layers)),
        }

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(output_dir / "tensorboard"),
            verbose=1,
            device="cpu",
        )

    print(f"Model device: {model.device}")
    print(f"Policy architecture: {model.policy}")
    print()

    # Build callback list
    callbacks = [checkpoint_callback, eval_callback, tensorboard_callback]

    # Train
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    print("\nSaving final model...")
    model.save(str(output_dir / "final_model"))
    env.save(str(output_dir / "vec_normalize.pkl"))

    print(f"\nTraining complete!")
    print(f"Models saved to: {output_dir}")
    print(f"\nTo visualize training progress:")
    print(f"  tensorboard --logdir {output_dir / 'tensorboard'}")

    env.close()
    eval_env.close()

    return str(output_dir)

