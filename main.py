#!/usr/bin/env python3
"""
Curve Following RL Agent
========================

A reinforcement learning agent that learns to follow curves using car-like dynamics.

Usage:
    # Train on sine curve
    python main.py train --curve-type sine

    # Train on random curves
    python main.py train --curve-type random --total-timesteps 1000000

    # Evaluate trained model
    python main.py eval --model-path outputs/sine_XXXXXX/final_model.zip

    # Demo with random agent (no training)
    python main.py demo --curve-type figure8

    # Interactive mode - test environment manually
    python main.py interactive --curve-type circle
"""

import argparse
import sys
from train import train
from evaluate import evaluate, demo_untrained
def cmd_train(args):
    """Train a new model."""
    
    train(args)


def cmd_eval(args):
    """Evaluate a trained model."""
    
    evaluate(args)


def cmd_demo(args):
    """Run demo with random agent."""
    demo_untrained(args)

def main():
    parser = argparse.ArgumentParser(
        description="Curve Following RL Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--curve-type", type=str, default="sine",
        choices=["sine", "arc", "scurve", "spiral", "hairpin", "random"],
    )
    train_parser.add_argument("--total-timesteps", type=int, default=500_000)
    train_parser.add_argument("--n-envs", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--n-steps", type=int, default=2048)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_parser.add_argument("--vf-coef", type=float, default=0.5)
    train_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    train_parser.add_argument("--output-dir", type=str, default="./outputs")
    train_parser.add_argument("--save-freq", type=int, default=10_000)
    train_parser.add_argument("--eval-freq", type=int, default=5_000)
    train_parser.add_argument("--resume", type=str, default=None, help="Path to model checkpoint to resume training from")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--model-path", type=str, required=True)
    eval_parser.add_argument(
        "--curve-type", type=str, default="sine",
        choices=["sine", "arc", "scurve", "spiral", "hairpin", "random"],
    )
    eval_parser.add_argument("--n-episodes", type=int, default=5)
    eval_parser.add_argument("--max-steps", type=int, default=1000)
    eval_parser.add_argument("--render", action="store_true", default=True)
    eval_parser.add_argument("--no-render", action="store_false", dest="render")
    eval_parser.add_argument("--deterministic", action="store_true", default=True)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Demo with random agent")
    demo_parser.add_argument(
        "--curve-type", type=str, default="sine",
        choices=["sine", "arc", "scurve", "spiral", "hairpin", "random"],
    )
    demo_parser.add_argument("--n-episodes", type=int, default=3)
    demo_parser.add_argument("--max-steps", type=int, default=500)

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive keyboard control"
    )
    interactive_parser.add_argument(
        "--curve-type", type=str, default="sine",
        choices=["sine", "arc", "scurve", "spiral", "hairpin", "random"],
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
