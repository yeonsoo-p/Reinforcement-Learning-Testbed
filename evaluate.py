#!/usr/bin/env python3
"""Evaluation and visualization script for trained models."""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.env import CurveFollowingEnv
from src.config import CURVE_PARAMS, RENDERER_CONFIG


def evaluate(args):
    """Evaluate a trained model with visualization."""
    model_path = Path(args.model_path)

    # Determine paths
    if model_path.is_dir():
        # Assume it's an output directory
        model_file = model_path / "final_model.zip"
        if not model_file.exists():
            model_file = model_path / "best_model" / "best_model.zip"
        vec_normalize_path = model_path / "vec_normalize.pkl"
    else:
        model_file = model_path
        vec_normalize_path = model_path.parent / "vec_normalize.pkl"

    print(f"Loading model from: {model_file}")

    # Curve parameters from config
    curve_kwargs = CURVE_PARAMS.get(args.curve_type)

    # Create environment
    env = CurveFollowingEnv(
        curve_type=args.curve_type,
        curve_kwargs=curve_kwargs,
        render_mode="human" if args.render else None,
        randomize_start=True,
        max_episode_steps=args.max_steps,
    )

    # Wrap in DummyVecEnv for compatibility
    vec_env = DummyVecEnv([lambda: env])

    # Load normalization statistics if available
    if vec_normalize_path.exists():
        print(f"Loading normalization stats from: {vec_normalize_path}")
        vec_env = VecNormalize.load(str(vec_normalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("Warning: No normalization stats found, using raw observations")

    # Load model
    model = PPO.load(str(model_file), env=vec_env, device="auto")
    print(f"Model loaded successfully (device: {model.device})")

    # Run evaluation episodes
    print(f"\nRunning {args.n_episodes} evaluation episodes...")
    print("-" * 50)

    episode_rewards = []
    episode_lengths = []
    episode_cross_track_errors = []

    for episode in range(args.n_episodes):
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        cross_track_errors = []

        # Reset renderer trail if visualizing
        if args.render and env._renderer is not None:
            env._renderer.reset_trail()

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=args.deterministic)

            # Step environment
            obs, reward, done, info = vec_env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            if "cross_track_error" in info[0]:
                cross_track_errors.append(abs(info[0]["cross_track_error"]))

            # Render
            if args.render:
                env.render()
                time.sleep(1.0 / RENDERER_CONFIG.render_fps)

            # Handle done
            if done[0]:
                break

        mean_cte = np.mean(cross_track_errors) if cross_track_errors else 0.0
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_cross_track_errors.append(mean_cte)

        # Determine outcome
        outcome = "truncated"
        if "reached_end" in info[0] and info[0]["reached_end"]:
            outcome = "COMPLETED"
        elif "off_track" in info[0] and info[0]["off_track"]:
            outcome = "off_track"

        print(f"Episode {episode + 1:3d}: "
              f"Reward = {episode_reward:8.2f}, "
              f"Length = {episode_length:5d}, "
              f"Mean CTE = {mean_cte:.3f}, "
              f"Outcome = {outcome}")

    # Summary statistics
    print("-" * 50)
    print(f"\nSummary over {args.n_episodes} episodes:")
    print(f"  Mean reward:     {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length:     {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Mean CTE:        {np.mean(episode_cross_track_errors):.3f} ± {np.std(episode_cross_track_errors):.3f}")
    print(f"  Completion rate: Episodes that reached the end of the curve")

    env.close()


def demo_untrained(args):
    """Run a demo with an untrained (random) agent."""
    print("Running demo with random agent...")
    print("This shows the environment before training.\n")

    # Curve parameters from config
    curve_kwargs = CURVE_PARAMS.get(args.curve_type)

    env = CurveFollowingEnv(
        curve_type=args.curve_type,
        curve_kwargs=curve_kwargs,
        render_mode="human",
        randomize_start=True,
        max_episode_steps=args.max_steps,
    )

    for episode in range(args.n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        if env._renderer is not None:
            env._renderer.reset_trail()

        print(f"\nEpisode {episode + 1}")

        while not done:
            # Random action
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            env.render()
            time.sleep(1.0 / RENDERER_CONFIG.render_fps)

        print(f"  Reward: {episode_reward:.2f}, Steps: {step}")
        if terminated:
            if info.get("reached_end", False):
                print("  COMPLETED: reached end of curve!")
            else:
                print("  Terminated: vehicle went off track")

    env.close()

