#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import panda_gym  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


def make_load_env(env_id: str) -> gym.Env:
    return Monitor(gym.make(env_id))


def make_render_env(env_id: str, renderer: str) -> gym.Env:
    return gym.make(env_id, render_mode="rgb_array", renderer=renderer)


def render_frame(env: gym.Env, width: int, height: int) -> np.ndarray:
    robot = env.unwrapped.robot
    return robot.sim.render(width=width, height=height)


def as_float_list(array: Any) -> list[float]:
    return np.asarray(array, dtype=np.float32).reshape(-1).tolist()


def build_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    observation = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)
    return np.concatenate([observation, achieved_goal, desired_goal], axis=0)


def pad_action_to_6d(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    padded = np.zeros(6, dtype=np.float32)
    padded[: action.shape[0]] = action
    return padded


def distance(obs: dict[str, np.ndarray]) -> float:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
    desired = np.asarray(obs["desired_goal"], dtype=np.float64)
    return float(np.linalg.norm(achieved - desired))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export successful visual demonstrations from a PandaPush SAC+HER teacher."
    )

    parser.add_argument("--env", default="PandaPush-v3")
    parser.add_argument("--model", required=True, type=Path)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/teacher_datasets/pandapush_sac_her_visual"),
    )

    parser.add_argument("--instruction", default="push the cube to the target")
    parser.add_argument("--target-successful-episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=150)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--min-steps", type=int, default=3)

    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--renderer", choices=["Tiny", "OpenGL"], default="Tiny")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    load_env = make_load_env(args.env)
    model = SAC.load(args.model, env=load_env, device=args.device)

    env = make_render_env(args.env, renderer=args.renderer)

    accepted_episode_summaries: list[dict[str, Any]] = []
    rejected_episode_summaries: list[dict[str, Any]] = []

    dataset_started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        attempt_idx = 0

        while (
            len(accepted_episode_summaries) < args.target_successful_episodes
            and attempt_idx < args.max_attempts
        ):
            episode_seed = args.seed + attempt_idx
            obs, info = env.reset(seed=episode_seed)

            episode_index = len(accepted_episode_summaries)
            temp_episode_dir = episodes_dir / f"attempt_{attempt_idx:06d}"
            images_dir = temp_episode_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            frame_records: list[dict[str, Any]] = []

            done = False
            step_idx = 0
            total_reward = 0.0

            initial_distance = distance(obs)
            best_distance = initial_distance
            final_distance = initial_distance
            final_success = float(info.get("is_success", False))

            while not done and step_idx < args.max_steps:
                frame = render_frame(env, args.width, args.height)
                image_rel_path = Path("images") / f"frame_{step_idx:06d}.png"
                image_abs_path = temp_episode_dir / image_rel_path
                imageio.imwrite(image_abs_path, frame)

                state = build_state(obs)

                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action_6d = pad_action_to_6d(action)

                next_obs, reward, terminated, truncated, info = env.step(action)

                next_distance = distance(next_obs)
                best_distance = min(best_distance, next_distance)
                final_distance = next_distance
                final_success = float(info.get("is_success", False))

                frame_record = {
                    "episode_index": episode_index,
                    "attempt_index": attempt_idx,
                    "step_index": step_idx,
                    "seed": episode_seed,
                    "instruction": args.instruction,
                    "image_path": str(image_rel_path),
                    "state": as_float_list(state),
                    "observation": as_float_list(obs["observation"]),
                    "achieved_goal": as_float_list(obs["achieved_goal"]),
                    "desired_goal": as_float_list(obs["desired_goal"]),
                    "action": as_float_list(action),
                    "action_6d": as_float_list(action_6d),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "is_success": final_success,
                    "distance_after_action": next_distance,
                }

                frame_records.append(frame_record)

                obs = next_obs
                total_reward += float(reward)
                step_idx += 1
                done = bool(terminated or truncated)

            accepted = bool(final_success >= 1.0 and step_idx >= args.min_steps)

            summary = {
                "accepted": accepted,
                "episode_index": episode_index if accepted else None,
                "attempt_index": attempt_idx,
                "seed": episode_seed,
                "steps": step_idx,
                "success": final_success,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "best_distance": best_distance,
                "total_reward": total_reward,
                "frames": len(frame_records),
            }

            if accepted:
                final_episode_dir = episodes_dir / f"episode_{episode_index:06d}"
                temp_episode_dir.rename(final_episode_dir)

                data_path = final_episode_dir / "data.jsonl"
                with data_path.open("w", encoding="utf-8") as f:
                    for record in frame_records:
                        f.write(json.dumps(record) + "\n")

                write_json(final_episode_dir / "summary.json", summary)
                accepted_episode_summaries.append(summary)

                print(
                    f"ACCEPTED episode={episode_index:06d} "
                    f"attempt={attempt_idx:06d} "
                    f"steps={step_idx} "
                    f"final_distance={final_distance:.6f}"
                )
            else:
                shutil.rmtree(temp_episode_dir)
                rejected_episode_summaries.append(summary)

                print(
                    f"REJECTED attempt={attempt_idx:06d} "
                    f"success={final_success} "
                    f"steps={step_idx} "
                    f"final_distance={final_distance:.6f}"
                )

            attempt_idx += 1

    finally:
        env.close()
        load_env.close()

    if not accepted_episode_summaries:
        raise RuntimeError("No successful episodes were exported.")

    total_frames = int(sum(ep["frames"] for ep in accepted_episode_summaries))
    success_rate_over_attempts = float(
        len(accepted_episode_summaries)
        / (len(accepted_episode_summaries) + len(rejected_episode_summaries))
    )

    metadata = {
        "dataset_name": args.output_dir.name,
        "created_at": dataset_started_at,
        "env_id": args.env,
        "teacher_model": str(args.model),
        "teacher_algorithm": "SAC+HER",
        "policy": "MultiInputPolicy",
        "instruction": args.instruction,
        "renderer": args.renderer,
        "image_width": args.width,
        "image_height": args.height,
        "target_successful_episodes": args.target_successful_episodes,
        "accepted_episodes": len(accepted_episode_summaries),
        "rejected_episodes": len(rejected_episode_summaries),
        "success_rate_over_attempts": success_rate_over_attempts,
        "total_frames": total_frames,
        "state_schema": "concat(observation[18], achieved_goal[3], desired_goal[3])",
        "state_dim": 24,
        "action_schema": "PandaPush-v3 real action [dx, dy, dz]",
        "action_dim": 3,
        "action_6d_schema": "real action padded to 6D for SmolVLA compatibility",
        "action_6d_dim": 6,
        "accepted_episode_summaries": accepted_episode_summaries,
        "rejected_episode_summaries": rejected_episode_summaries,
    }

    write_json(args.output_dir / "metadata.json", metadata)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
