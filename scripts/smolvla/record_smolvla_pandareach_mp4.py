from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

try:
    import panda_gym  # noqa: F401
except Exception:
    pass

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pandareach_smolvla_utils import (
    SmolVLAPandaReachPolicy,
    distance,
    normalize_instruction,
)


DEFAULT_CHECKPOINT = "outputs/train/smolvla_panda_reach_chunk1_500/checkpoints/000400/pretrained_model"
DEFAULT_DATASET_ROOT = "outputs/lerobot_datasets/panda_reach_ppo_teacher_50"
DEFAULT_REPO_ID = "local/panda_reach_ppo_teacher_50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a natural-looking SmolVLA PandaReach-v3 MP4."
    )

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)

    parser.add_argument("--instruction", default=None)
    parser.add_argument("--ask-instruction", action="store_true")

    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-json", default=None)

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--ema", type=float, default=0.85)
    parser.add_argument("--hold", type=int, default=60)
    parser.add_argument("--initial-hold", type=int, default=60)
    parser.add_argument("--success-hold", type=int, default=90)
    parser.add_argument(
        "--visual-substeps",
        type=int,
        default=30,
        help="Number of PyBullet substeps rendered for each SmolVLA action.",
    )

    parser.add_argument("--renderer", choices=["Tiny", "OpenGL"], default="Tiny")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    return parser.parse_args()


def safe_makedirs_for_file(path: str | Path) -> None:
    directory = Path(path).parent
    if str(directory):
        directory.mkdir(parents=True, exist_ok=True)


def validate_paths(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    dataset_root = Path(args.dataset_root)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device_arg


def make_env(renderer: str):
    return gym.make(
        "PandaReach-v3",
        render_mode="rgb_array",
        renderer=renderer,
    )


def render_frame(env, width: int, height: int) -> np.ndarray:
    robot = env.unwrapped.robot
    return robot.sim.render(width=width, height=height)


def append_hold_frames(writer, frame: np.ndarray, count: int) -> None:
    for _ in range(max(0, int(count))):
        writer.append_data(frame)


def smooth_action(
    action: np.ndarray,
    prev_action: np.ndarray | None,
    ema: float,
) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)

    if ema <= 0.0:
        return action, action.copy()

    if prev_action is None:
        prev_action = action.copy()

    smoothed = float(ema) * prev_action + (1.0 - float(ema)) * action
    return smoothed.astype(np.float32), smoothed.astype(np.float32)


def visual_policy_step(env, action: np.ndarray, writer, width: int, height: int, visual_substeps: int):
    base_env = env.unwrapped

    base_env.robot.set_action(action)

    for _ in range(max(1, int(visual_substeps))):
        base_env.sim.physics_client.stepSimulation()
        frame = render_frame(env, width, height)
        writer.append_data(frame)

    obs = base_env._get_obs()
    info = {
        "is_success": bool(
            base_env.task.is_success(
                obs["achieved_goal"],
                base_env.task.get_goal(),
            )
        )
    }
    reward = float(
        base_env.task.compute_reward(
            obs["achieved_goal"],
            base_env.task.get_goal(),
            info,
        )
    )
    terminated = bool(info["is_success"])
    truncated = False

    return obs, reward, terminated, truncated, info


def run_episode(
    env,
    policy: SmolVLAPandaReachPolicy,
    writer,
    *,
    episode_index: int,
    seed: int,
    steps: int,
    instruction: str,
    action_scale: float,
    ema: float,
    hold: int,
    initial_hold: int,
    success_hold: int,
    width: int,
    height: int,
    visual_substeps: int,
) -> dict:
    obs, info = env.reset(seed=seed)

    initial_distance = distance(obs)
    best_distance = initial_distance
    final_distance = initial_distance
    success_seen = bool(info.get("is_success", False))

    first_frame = render_frame(env, width, height)
    writer.append_data(first_frame)
    append_hold_frames(writer, first_frame, initial_hold)

    prev_action = None
    trajectory = []

    for step in range(int(steps)):
        rgb = env.render()
        if rgb is None:
            raise RuntimeError("env.render() returned None.")

        raw_action = policy.act(obs, rgb, task=instruction)
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)[:3]

        smoothed_action, prev_action = smooth_action(raw_action, prev_action, ema)
        action = smoothed_action * float(action_scale)
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

        obs, reward, terminated, truncated, info = visual_policy_step(
            env=env,
            action=action,
            writer=writer,
            width=width,
            height=height,
            visual_substeps=visual_substeps,
        )

        current_distance = distance(obs)
        best_distance = min(best_distance, current_distance)
        final_distance = current_distance
        is_success = bool(info.get("is_success", False))
        success_seen = success_seen or is_success

        frame = render_frame(env, width, height)
        writer.append_data(frame)

        trajectory.append(
            {
                "step": int(step + 1),
                "reward": float(reward),
                "distance": float(current_distance),
                "is_success": bool(is_success),
                "raw_action": raw_action.tolist(),
                "smoothed_action": smoothed_action.tolist(),
                "executed_action": action.tolist(),
            }
        )

        if terminated or truncated or is_success:
            append_hold_frames(writer, frame, success_hold)
            break

    last_frame = render_frame(env, width, height)
    append_hold_frames(writer, last_frame, hold)

    return {
        "episode": int(episode_index),
        "seed": int(seed),
        "steps": int(len(trajectory)),
        "initial_distance": float(initial_distance),
        "best_distance": float(best_distance),
        "final_distance": float(final_distance),
        "distance_improvement": float(initial_distance - best_distance),
        "success_seen": bool(success_seen),
        "trajectory": trajectory,
    }


def main() -> None:
    args = parse_args()
    validate_paths(args)

    if args.ask_instruction:
        raw_instruction = input("Instruction [reach the target]: ").strip() or "reach the target"
    elif args.instruction is not None:
        raw_instruction = args.instruction
    else:
        raw_instruction = "reach the target"

    instruction = normalize_instruction(raw_instruction)

    device = resolve_device(args.device)

    safe_makedirs_for_file(args.out)
    if args.summary_json is not None:
        safe_makedirs_for_file(args.summary_json)

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset_root: {args.dataset_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"instruction: {raw_instruction}")
    print(f"mapped_instruction: {instruction}")
    print(f"seed: {args.seed}")
    print(f"episodes: {args.episodes}")
    print(f"action_scale: {args.action_scale}")
    print(f"ema: {args.ema}")
    print(f"fps: {args.fps}")
    print(f"out: {args.out}")

    print("\nLoading LeRobot dataset statistics...")
    dataset = LeRobotDataset(args.repo_id, root=Path(args.dataset_root))
    dataset_stats = dataset.meta.stats

    print("Loading SmolVLA checkpoint...")
    policy = SmolVLAPandaReachPolicy(args.checkpoint, dataset_stats, device)

    print("Opening PandaReach-v3 renderer...")
    env = make_env(args.renderer)

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)

    episodes = []

    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            print(f"\n[EP {episode_index + 1:02d}/{args.episodes}] seed={episode_seed}")

            episode = run_episode(
                env,
                policy,
                writer,
                episode_index=episode_index,
                seed=episode_seed,
                steps=args.steps,
                instruction=instruction,
                action_scale=args.action_scale,
                ema=args.ema,
                hold=args.hold,
                initial_hold=args.initial_hold,
                success_hold=args.success_hold,
                width=args.width,
                height=args.height,
                visual_substeps=args.visual_substeps,
            )

            episodes.append(episode)

            print(
                f"steps={episode['steps']} "
                f"success={episode['success_seen']} "
                f"initial={episode['initial_distance']:.4f} "
                f"best={episode['best_distance']:.4f} "
                f"final={episode['final_distance']:.4f}"
            )

    finally:
        writer.close()
        policy.close()
        env.close()

    summary = {
        "ok": True,
        "purpose": "SmolVLA visual PandaReach-v3 MP4 demo",
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "repo_id": args.repo_id,
        "raw_instruction": raw_instruction,
        "mapped_instruction": instruction,
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "steps_limit": int(args.steps),
        "fps": int(args.fps),
        "width": int(args.width),
        "height": int(args.height),
        "action_scale": float(args.action_scale),
        "ema": float(args.ema),
        "hold": int(args.hold),
        "initial_hold": int(args.initial_hold),
        "success_hold": int(args.success_hold),
        "visual_substeps": int(args.visual_substeps),
        "renderer": args.renderer,
        "device": device,
        "video_path": args.out,
        "episodes_detail": episodes,
    }

    if args.summary_json is not None:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved summary: {args.summary_json}")

    print(f"\n[OK] wrote video: {args.out}")
    print(json.dumps(
        {
            "success_rate": float(np.mean([ep["success_seen"] for ep in episodes])),
            "mean_initial_distance": float(np.mean([ep["initial_distance"] for ep in episodes])),
            "mean_best_distance": float(np.mean([ep["best_distance"] for ep in episodes])),
            "mean_steps": float(np.mean([ep["steps"] for ep in episodes])),
            "video_path": args.out,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
