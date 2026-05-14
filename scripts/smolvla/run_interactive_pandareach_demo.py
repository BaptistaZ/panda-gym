from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    import panda_gym  # noqa: F401
except Exception:
    pass

import gymnasium as gym

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
        description="Interactive visual SmolVLA demo for PandaReach-v3."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--pre-delay", type=float, default=1.5)
    parser.add_argument("--post-delay", type=float, default=8.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--render-width", type=int, default=720)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--renderer", choices=["OpenGL", "Tiny"], default="OpenGL")
    parser.add_argument("--countdown", type=int, default=3)

    parser.add_argument(
        "--mode",
        choices=["single-action", "closed-loop"],
        default="single-action",
        help="single-action computes one SmolVLA action and visualizes it smoothly; closed-loop recomputes actions until success.",
    )
    parser.add_argument(
        "--visual-substeps",
        type=int,
        default=120,
        help="Number of PyBullet simulation substeps used to visualize each action smoothly.",
    )
    parser.add_argument(
        "--visual-substep-delay",
        type=float,
        default=0.01,
        help="Real-time delay between visual simulation substeps.",
    )
    parser.add_argument(
        "--verbose-steps",
        action="store_true",
        help="Print every policy step in closed-loop mode.",
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    dataset_root = Path(args.dataset_root)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")


def make_visual_env(args: argparse.Namespace):
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

    end_time = time.time() + seconds
    while time.time() < end_time:
        env.render()
        time.sleep(0.03)


def terminal_countdown(env, seconds: int) -> None:
    if seconds <= 0:
        return

    for value in range(seconds, 0, -1):
        print(f"Executing in {value}...")
        wait_with_render(env, 1.0)


def read_instruction_from_terminal(env, args: argparse.Namespace) -> tuple[str, str]:
    print("\nInteractive instruction mode")
    print("Accepted examples:")
    print("  - reach the target")
    print("  - alcança o alvo")
    print("  - chega ao alvo")
    print("\nThe PyBullet window is already open. The robot is waiting for your instruction.")

    if args.instruction is not None:
        raw_instruction = args.instruction
        print(f"\nInstruction: {raw_instruction}")
    else:
        raw_instruction = input("\nInstruction [reach the target]: ").strip() or "reach the target"

    task = normalize_instruction(raw_instruction)
    print(f"Mapped instruction: {task}")

    terminal_countdown(env, args.countdown)

    return raw_instruction, task


def visual_env_step(
    env,
    action: np.ndarray,
    visual_substeps: int,
    visual_substep_delay: float,
):
    base_env = env.unwrapped

    base_env.robot.set_action(action)

    for _ in range(max(1, int(visual_substeps))):
        base_env.sim.physics_client.stepSimulation()
        env.render()

        if visual_substep_delay > 0:
            time.sleep(visual_substep_delay)

    obs = base_env._get_obs()
    info = {
        "is_success": bool(
            base_env.task.is_success(
                obs["achieved_goal"],
                base_env.task.get_goal(),
            )
        )
    }
    terminated = bool(info["is_success"])
    truncated = False
    reward = float(
        base_env.task.compute_reward(
            obs["achieved_goal"],
            base_env.task.get_goal(),
            info,
        )
    )

    return obs, reward, terminated, truncated, info


def compute_smolvla_action(
    policy: SmolVLAPandaReachPolicy,
    env,
    obs: dict,
    task: str,
    action_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = env.render()
    if rgb is None:
        raise RuntimeError("env.render() returned None. Use render_mode='rgb_array'.")

    raw_action = policy.act(obs, rgb, task=task)
    scaled_action = raw_action[:3] * float(action_scale)
    action = np.clip(scaled_action, env.action_space.low, env.action_space.high)

    return raw_action, action


def run_demo(args: argparse.Namespace) -> dict:
    validate_paths(args)

    raw_instruction = ""
    task = "reach the target"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset_root: {args.dataset_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"seed: {args.seed}")
    print(f"mode: {args.mode}")
    print(f"action_scale: {args.action_scale}")
    print(f"visual_substeps: {args.visual_substeps}")
    print(f"visual_substep_delay: {args.visual_substep_delay}")

    print("\nLoading LeRobot dataset statistics...")
    dataset = LeRobotDataset(args.repo_id, root=Path(args.dataset_root))
    dataset_stats = dataset.meta.stats

    print("Loading SmolVLA policy...")
    policy = SmolVLAPandaReachPolicy(args.checkpoint, dataset_stats, device)

    print("Opening PandaReach-v3 visual environment...")
    env = make_visual_env(args)

    obs, info = env.reset(seed=args.seed)
    initial_distance = distance(obs)
    best_distance = initial_distance
    final_distance = initial_distance
    success = bool(info.get("is_success", False))
    steps = 0

    try:
        env.render()

        print("\nInitial scene ready.")
        print(f"initial_distance: {initial_distance:.4f}")
        print("Robot and target are visible in the PyBullet window.")
        wait_with_render(env, args.pre_delay)

        raw_instruction, task = read_instruction_from_terminal(env, args)

        if args.mode == "single-action":
            print("\nComputing one SmolVLA action...")
            raw_action, action = compute_smolvla_action(
                policy=policy,
                env=env,
                obs=obs,
                task=task,
                action_scale=args.action_scale,
            )

            print("Executing one natural visual motion...")
            print(f"raw_action: {raw_action[:3].tolist()}")
            print(f"executed_action: {action.tolist()}")

            obs, reward, terminated, truncated, info = visual_env_step(
                env=env,
                action=action,
                visual_substeps=args.visual_substeps,
                visual_substep_delay=args.visual_substep_delay,
            )

            steps = 1
            final_distance = distance(obs)
            best_distance = min(best_distance, final_distance)
            success = bool(info.get("is_success", False))

        else:
            print("\nExecuting SmolVLA closed-loop policy...")

            for step in range(args.max_steps):
                raw_action, action = compute_smolvla_action(
                    policy=policy,
                    env=env,
                    obs=obs,
                    task=task,
                    action_scale=args.action_scale,
                )

                obs, reward, terminated, truncated, info = visual_env_step(
                    env=env,
                    action=action,
                    visual_substeps=args.visual_substeps,
                    visual_substep_delay=args.visual_substep_delay,
                )

                current_distance = distance(obs)
                best_distance = min(best_distance, current_distance)
                final_distance = current_distance
                success = success or bool(info.get("is_success", False))
                steps = step + 1

                if args.verbose_steps:
                    print(
                        f"step={steps:02d} "
                        f"distance={current_distance:.4f} "
                        f"best={best_distance:.4f} "
                        f"success={bool(info.get('is_success', False))} "
                        f"action={action.tolist()}"
                    )

                if terminated or truncated or bool(info.get("is_success", False)):
                    break

        print("\nExecution finished.")
        print(f"success: {success}")
        print(f"steps: {steps}")
        print(f"initial_distance: {initial_distance:.4f}")
        print(f"best_distance: {best_distance:.4f}")
        print(f"final_distance: {final_distance:.4f}")
        print("Keeping the final scene visible...")

        wait_with_render(env, args.post_delay)

    finally:
        policy.close()
        env.close()

    return {
        "success": bool(success),
        "steps": int(steps),
        "initial_distance": float(initial_distance),
        "best_distance": float(best_distance),
        "final_distance": float(final_distance),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "repo_id": args.repo_id,
        "instruction": raw_instruction,
        "mapped_task": task,
        "mode": args.mode,
        "action_scale": float(args.action_scale),
        "visual_substeps": int(args.visual_substeps),
        "visual_substep_delay": float(args.visual_substep_delay),
    }


def main() -> None:
    args = parse_args()
    result = run_demo(args)

    print("\nDemo result:")
    print(json.dumps(result, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()
