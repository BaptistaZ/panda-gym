from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
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
        description="Final live SmolVLA PandaReach-v3 demo with natural PyBullet motion."
    )

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=50)

    parser.add_argument("--action-scale", type=float, default=0.35)
    parser.add_argument("--command-ema", type=float, default=0.0)

    parser.add_argument("--frame-delay", type=float, default=0.008)
    parser.add_argument("--pre-delay", type=float, default=2.0)
    parser.add_argument("--post-delay", type=float, default=8.0)
    parser.add_argument("--countdown", type=int, default=3)

    parser.add_argument("--worse-margin", type=float, default=0.025)
    parser.add_argument("--worse-patience", type=int, default=3)

    parser.add_argument("--render-width", type=int, default=720)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--renderer", choices=["OpenGL", "Tiny"], default="OpenGL")

    parser.add_argument("--instruction", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not Path(args.dataset_root).exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")


def make_env(args: argparse.Namespace):
    return gym.make(
        "PandaReach-v3",
        render_mode="rgb_array",
        renderer=args.renderer,
        render_width=args.render_width,
        render_height=args.render_height,
    )


def wait_with_render(env, seconds: float) -> None:
    if seconds <= 0:
        return

    end = time.time() + seconds
    while time.time() < end:
        env.render()
        time.sleep(0.03)


def countdown(env, seconds: int) -> None:
    for value in range(seconds, 0, -1):
        print(f"Executing in {value}...")
        wait_with_render(env, 1.0)


def read_instruction(args: argparse.Namespace) -> tuple[str, str]:
    print("\nInteractive instruction mode")
    print("Accepted examples:")
    print("  - reach the target")
    print("  - alcança o alvo")
    print("  - chega ao alvo")

    if args.instruction is not None:
        raw_instruction = args.instruction
        print(f"\nInstruction: {raw_instruction}")
    else:
        raw_instruction = input("\nInstruction [reach the target]: ").strip() or "reach the target"

    mapped_instruction = normalize_instruction(raw_instruction)
    print(f"Mapped instruction: {mapped_instruction}")

    return raw_instruction, mapped_instruction


def smooth_command(
    action: np.ndarray,
    previous_action: np.ndarray | None,
    command_ema: float,
) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)[:3]

    if command_ema <= 0.0 or previous_action is None:
        return action, action.copy()

    smoothed = float(command_ema) * previous_action + (1.0 - float(command_ema)) * action
    return smoothed.astype(np.float32), smoothed.astype(np.float32)


def exact_env_step_with_render(env, action: np.ndarray, frame_delay: float):
    base_env = env.unwrapped

    base_env.robot.set_action(action)

    # This matches env.step(): it uses the environment's own PyBullet substep count.
    for _ in range(int(base_env.sim.n_substeps)):
        base_env.sim.physics_client.stepSimulation()
        env.render()

        if frame_delay > 0:
            time.sleep(frame_delay)

    obs = base_env._get_obs()

    terminated = bool(
        base_env.task.is_success(
            obs["achieved_goal"],
            base_env.task.get_goal(),
        )
    )

    info = {"is_success": terminated}

    reward = float(
        base_env.task.compute_reward(
            obs["achieved_goal"],
            base_env.task.get_goal(),
            info,
        )
    )

    truncated = False

    return obs, reward, terminated, truncated, info


def main() -> None:
    args = parse_args()
    validate_paths(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Final SmolVLA PandaReach live demo")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset_root: {args.dataset_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"seed: {args.seed}")
    print(f"action_scale: {args.action_scale}")
    print(f"command_ema: {args.command_ema}")
    print(f"frame_delay: {args.frame_delay}")

    print("\nLoading LeRobot dataset statistics...")
    dataset = LeRobotDataset(args.repo_id, root=Path(args.dataset_root))
    dataset_stats = dataset.meta.stats

    print("Loading SmolVLA checkpoint...")
    policy = SmolVLAPandaReachPolicy(args.checkpoint, dataset_stats, device)

    print("Opening PandaReach-v3 world...")
    env = make_env(args)

    obs, info = env.reset(seed=args.seed)

    initial_distance = distance(obs)
    best_distance = initial_distance
    final_distance = initial_distance
    success = bool(info.get("is_success", False))

    raw_instruction = ""
    mapped_instruction = "reach the target"

    previous_action = None
    steps = 0
    worse_count = 0

    try:
        env.render()

        print("\nInitial scene ready.")
        print(f"initial_distance: {initial_distance:.4f}")
        print("The robot and target are stopped in the PyBullet window.")
        wait_with_render(env, args.pre_delay)

        raw_instruction, mapped_instruction = read_instruction(args)

        print("\nInstruction received. The robot will now execute the SmolVLA policy.")
        countdown(env, args.countdown)

        print("\nExecuting final SmolVLA motion...")

        for step in range(args.max_policy_steps):
            rgb = env.render()
            if rgb is None:
                raise RuntimeError("env.render() returned None.")

            raw_action = policy.act(obs, rgb, task=mapped_instruction)
            smoothed_action, previous_action = smooth_command(
                action=raw_action,
                previous_action=previous_action,
                command_ema=args.command_ema,
            )

            executed_action = smoothed_action * float(args.action_scale)
            executed_action = np.clip(
                executed_action,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)

            obs, reward, terminated, truncated, info = exact_env_step_with_render(
                env=env,
                action=executed_action,
                frame_delay=args.frame_delay,
            )

            current_distance = distance(obs)
            final_distance = current_distance
            success = success or bool(info.get("is_success", False))
            steps = step + 1

            if current_distance < best_distance:
                best_distance = current_distance
                worse_count = 0
            elif current_distance > best_distance + float(args.worse_margin):
                worse_count += 1

            if args.verbose:
                print(
                    f"policy_update={steps:03d} "
                    f"distance={current_distance:.4f} "
                    f"best={best_distance:.4f} "
                    f"success={bool(info.get('is_success', False))}"
                )

            if terminated or truncated or bool(info.get("is_success", False)):
                break

            if worse_count >= int(args.worse_patience):
                print(
                    "\nStopping early because the robot started moving away from the target."
                )
                break

        print("\nExecution finished.")
        print(f"success: {success}")
        print(f"policy_updates: {steps}")
        print(f"initial_distance: {initial_distance:.4f}")
        print(f"best_distance: {best_distance:.4f}")
        print(f"final_distance: {final_distance:.4f}")
        print("Keeping the final scene visible...")

        wait_with_render(env, args.post_delay)

    finally:
        policy.close()
        env.close()

    result = {
        "success": bool(success),
        "policy_updates": int(steps),
        "initial_distance": float(initial_distance),
        "best_distance": float(best_distance),
        "final_distance": float(final_distance),
        "seed": int(args.seed),
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "repo_id": args.repo_id,
        "instruction": raw_instruction,
        "mapped_instruction": mapped_instruction,
        "action_scale": float(args.action_scale),
        "command_ema": float(args.command_ema),
        "frame_delay": float(args.frame_delay),
        "worse_margin": float(args.worse_margin),
        "worse_patience": int(args.worse_patience),
    }

    print("\nDemo result:")
    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()
