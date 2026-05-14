#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


def make_env(env_id: str) -> gym.Env:
    return Monitor(gym.make(env_id))


def evaluate_model(
    model: SAC,
    env_id: str,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> dict[str, Any]:
    env = make_env(env_id)

    successes: list[float] = []
    final_distances: list[float] = []
    best_distances: list[float] = []
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []

    max_steps = env.spec.max_episode_steps if env.spec is not None else 50

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed + episode_idx)

        done = False
        step_count = 0
        total_reward = 0.0

        achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
        desired = np.asarray(obs["desired_goal"], dtype=np.float64)

        best_distance = float(np.linalg.norm(achieved - desired))
        final_distance = best_distance
        final_success = float(info.get("is_success", False))

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
            desired = np.asarray(obs["desired_goal"], dtype=np.float64)

            final_distance = float(np.linalg.norm(achieved - desired))
            best_distance = min(best_distance, final_distance)
            final_success = float(info.get("is_success", False))

            total_reward += float(reward)
            step_count += 1
            done = bool(terminated or truncated)

        successes.append(final_success)
        final_distances.append(final_distance)
        best_distances.append(best_distance)
        episode_lengths.append(step_count)
        episode_rewards.append(total_reward)

    env.close()

    return {
        "env_id": env_id,
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "success_rate": float(np.mean(successes)),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_best_distance": float(np.mean(best_distances)),
        "max_final_distance": float(np.max(final_distances)),
        "min_final_distance": float(np.min(final_distances)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "successes": successes,
        "final_distances": final_distances,
        "best_distances": best_distances,
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PandaPush SAC+HER teacher model.")
    parser.add_argument("--env", default="PandaPush-v3")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_env = make_env(args.env)
    model = SAC.load(args.model, env=load_env)

    metrics = evaluate_model(
        model=model,
        env_id=args.env,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
    )

    metrics["model_path"] = str(args.model)

    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
