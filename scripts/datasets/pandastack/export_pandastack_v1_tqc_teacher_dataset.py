#!/usr/bin/env python3
"""
Export raw rollouts from the official sb3/tqc-PandaStack-v1 teacher.

Important:
    This script must be executed in the legacy Conda environment:
        panda-stack-v1-exact

    It depends on:
        Python 3.8.20
        gym 0.21.0
        panda-gym 1.1.1
        stable-baselines3 1.8.0
        sb3-contrib 1.8.0

    Do not run this script inside the main panda-smolvla environment.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import gym
import numpy as np
import panda_gym  # noqa: F401
from PIL import Image
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper


def as_float_list(value):
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def distance(obs):
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64).reshape(-1)
    desired = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(achieved - desired))


def build_state(obs):
    observation = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)
    return np.concatenate([observation, achieved_goal, desired_goal], axis=0)


def pad_action_to_6d(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    padded = np.zeros(6, dtype=np.float32)
    padded[: action.shape[0]] = action
    return padded


def make_env(env_id):
    env = gym.make(env_id)
    env = TimeFeatureWrapper(env)
    return env


def render_frame(env, width, height):
    errors = []

    try:
        frame = env.unwrapped.robot.sim.render(
            mode="rgb_array",
            width=width,
            height=height,
        )
        if frame is not None:
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            return frame
    except Exception as exc:
        errors.append(f"sim.render failed: {type(exc).__name__}: {exc}")

    try:
        frame = env.render(
            mode="rgb_array",
            width=width,
            height=height,
        )
        if frame is not None:
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            return frame
    except TypeError:
        try:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                return np.asarray(frame, dtype=np.uint8)
        except Exception as exc:
            errors.append(f"env.render fallback failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"env.render failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not render RGB frame from PandaStack-v1. " + " | ".join(errors))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export raw rollouts from the official sb3/tqc-PandaStack-v1 teacher."
    )
    parser.add_argument("--env", default="PandaStack-v1")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("logs/tqc/PandaStack-v1_1/tqc-PandaStack-v1.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/teacher_datasets/pandastack_v1_tqc_raw_20"),
    )
    parser.add_argument("--instruction", default="stack the blocks")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env)

    model = TQC.load(
        str(args.model),
        env=env,
        device=args.device,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
        },
    )

    metadata = {
        "dataset_name": args.output_dir.name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env_id": args.env,
        "teacher": "sb3/tqc-PandaStack-v1",
        "teacher_algorithm": "TQC",
        "model_path": str(args.model),
        "wrapper": "sb3_contrib.common.wrappers.TimeFeatureWrapper",
        "instruction": args.instruction,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed_start": args.seed,
        "state_definition": "concat(observation, achieved_goal, desired_goal)",
        "expected_state_shape": 44,
        "raw_action_shape": 4,
        "stored_action_shape": 6,
        "note": (
            "This dataset intentionally stores all rollouts, not only successful episodes, "
            "because the official teacher showed weak local performance."
        ),
    }

    episode_summaries = []
    total_frames = 0

    try:
        for episode_idx in range(args.episodes):
            episode_seed = args.seed + episode_idx
            env.seed(episode_seed)
            obs = env.reset()

            episode_dir = episodes_dir / f"episode_{episode_idx:06d}"
            images_dir = episode_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            records = []

            initial_distance = distance(obs)
            best_distance = initial_distance
            final_distance = initial_distance
            success_seen = False
            total_reward = 0.0

            for step_idx in range(args.max_steps):
                frame = render_frame(env, args.width, args.height)
                image_rel_path = Path("images") / f"frame_{step_idx:06d}.png"
                image_abs_path = episode_dir / image_rel_path
                Image.fromarray(frame).save(image_abs_path)

                state = build_state(obs)

                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                action_6d = pad_action_to_6d(action)

                next_obs, reward, done, info = env.step(action)

                next_distance = distance(next_obs)
                best_distance = min(best_distance, next_distance)
                final_distance = next_distance
                is_success = bool(info.get("is_success", False))
                success_seen = success_seen or is_success
                total_reward += float(reward)

                records.append({
                    "episode_index": episode_idx,
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
                    "done": bool(done),
                    "is_success": is_success,
                    "distance_after_action": next_distance,
                })

                obs = next_obs
                total_frames += 1

                if done:
                    break

            data_path = episode_dir / "data.jsonl"
            with data_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            summary = {
                "episode_index": episode_idx,
                "seed": episode_seed,
                "frames": len(records),
                "initial_distance": initial_distance,
                "best_distance": best_distance,
                "final_distance": final_distance,
                "improvement": initial_distance - best_distance,
                "success_seen": success_seen,
                "total_reward": total_reward,
            }

            write_json(episode_dir / "summary.json", summary)
            episode_summaries.append(summary)

            print(
                f"EXPORTED episode={episode_idx:06d} "
                f"frames={len(records)} "
                f"success={success_seen} "
                f"initial_distance={initial_distance:.6f} "
                f"best_distance={best_distance:.6f} "
                f"final_distance={final_distance:.6f}"
            )

    finally:
        env.close()

    metadata["total_frames"] = total_frames
    metadata["success_rate"] = float(np.mean([ep["success_seen"] for ep in episode_summaries]))
    metadata["mean_initial_distance"] = float(np.mean([ep["initial_distance"] for ep in episode_summaries]))
    metadata["mean_best_distance"] = float(np.mean([ep["best_distance"] for ep in episode_summaries]))
    metadata["mean_final_distance"] = float(np.mean([ep["final_distance"] for ep in episode_summaries]))
    metadata["mean_improvement"] = float(np.mean([ep["improvement"] for ep in episode_summaries]))
    metadata["episode_summaries"] = episode_summaries

    write_json(args.output_dir / "metadata.json", metadata)

    print("EXPORT_OK")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "episodes": args.episodes,
        "total_frames": total_frames,
        "success_rate": metadata["success_rate"],
        "mean_improvement": metadata["mean_improvement"],
    }, indent=2))


if __name__ == "__main__":
    main()