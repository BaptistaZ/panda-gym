#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


def make_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    if render_mode is None:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode=render_mode)
    env = Monitor(env)
    return env


def evaluate_model(
    model: SAC,
    env_id: str,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    env = make_env(env_id)

    successes: list[float] = []
    final_distances: list[float] = []
    best_distances: list[float] = []
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []

    max_steps = env.spec.max_episode_steps if env.spec is not None else 100

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
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_final_distance": float(np.mean(final_distances)) if final_distances else None,
        "mean_best_distance": float(np.mean(best_distances)) if best_distances else None,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else None,
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else None,
        "successes": successes,
        "final_distances": final_distances,
        "best_distances": best_distances,
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SAC+HER teacher policy for PandaPush-v3."
    )

    parser.add_argument("--env", default="PandaPush-v3")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n-sampled-goal", type=int, default=4)

    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/PandaPush/ee/sac_her"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = (
        f"{args.env.lower()}_sac_her"
        f"_steps{args.total_timesteps}"
        f"_seed{args.seed}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    run_dir = args.output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["run_dir"] = str(run_dir)
    save_json(run_dir / "config.json", config)

    env = make_env(args.env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": "future",
        },
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,
    )

    best_success_rate = -1.0
    trained_steps = 0

    while trained_steps < args.total_timesteps:
        next_chunk = min(args.eval_every, args.total_timesteps - trained_steps)

        model.learn(
            total_timesteps=next_chunk,
            reset_num_timesteps=False,
            tb_log_name="sac_her",
            progress_bar=True,
        )

        trained_steps += next_chunk

        checkpoint_path = checkpoint_dir / f"step_{trained_steps:08d}.zip"
        model.save(checkpoint_path)

        metrics = evaluate_model(
            model=model,
            env_id=args.env,
            episodes=args.eval_episodes,
            seed=args.seed + trained_steps,
            deterministic=True,
        )
        metrics["trained_steps"] = trained_steps
        metrics["checkpoint_path"] = str(checkpoint_path)

        metrics_path = metrics_dir / f"eval_step_{trained_steps:08d}.json"
        save_json(metrics_path, metrics)

        print(json.dumps(metrics, indent=2))

        if metrics["success_rate"] > best_success_rate:
            best_success_rate = metrics["success_rate"]
            model.save(run_dir / "best.zip")
            save_json(run_dir / "best_metrics.json", metrics)

    model.save(run_dir / "final.zip")

    final_metrics = evaluate_model(
        model=model,
        env_id=args.env,
        episodes=max(args.eval_episodes, 50),
        seed=args.seed + 10_000,
        deterministic=True,
    )
    final_metrics["trained_steps"] = trained_steps
    final_metrics["model_path"] = str(run_dir / "final.zip")
    save_json(run_dir / "final_metrics.json", final_metrics)

    env.close()

    print("Training complete.")
    print(f"Run directory: {run_dir}")
    print(f"Best success rate: {best_success_rate}")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
