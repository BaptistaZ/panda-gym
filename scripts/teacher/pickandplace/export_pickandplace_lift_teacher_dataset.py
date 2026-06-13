from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
from PIL import Image
from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


TASK_TEXT = "pick up the cube"
OBJECT_Z = 0.02
HOVER_Z = 0.12
TEACHER_MAX_STEPS = 35
DEFAULT_MIN_LIFT_HEIGHT = 0.03


def make_normalization_env():
    """Create the environment required for loading VecNormalize and TQC."""
    return gym.make(
        "PandaPickAndPlace-v3",
        render_mode="rgb_array",
        renderer="Tiny",
    )


def make_eval_env():
    """Create the raw PandaPickAndPlace environment."""
    return PandaPickAndPlaceEnv(
        render_mode="rgb_array",
        renderer="Tiny",
    )


def build_state_25(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Build the 25-dimensional state used by the existing dataset."""
    state = np.concatenate(
        [
            np.asarray(observation["observation"], dtype=np.float32),
            np.asarray(observation["achieved_goal"], dtype=np.float32),
            np.asarray(observation["desired_goal"], dtype=np.float32),
        ]
    ).astype(np.float32)

    if state.shape != (25,):
        raise ValueError(f"Expected state shape (25,), got {state.shape}")

    return state


def action_4d_to_smolvla_6d(action_4d: np.ndarray) -> np.ndarray:
    """Pad the Panda action to the six dimensions expected by SmolVLA."""
    action_4d = np.asarray(action_4d, dtype=np.float32).reshape(4)
    action_6d = np.zeros(6, dtype=np.float32)
    action_6d[:4] = action_4d
    return action_6d


def save_rgb_frame(frame: np.ndarray, path: Path) -> None:
    """Save a rendered RGB frame as PNG."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    Image.fromarray(frame).save(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write formatted JSON."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def predict_teacher(
    model: TQC,
    normalizer: VecNormalize,
    observation: dict[str, np.ndarray],
    proxy_goal: np.ndarray,
) -> np.ndarray:
    """Predict the RL teacher action for an elevated proxy goal."""
    batch = {
        key: np.expand_dims(np.asarray(value), axis=0)
        for key, value in observation.items()
    }

    batch["desired_goal"] = np.expand_dims(
        np.asarray(proxy_goal, dtype=np.float32),
        axis=0,
    )

    normalized_observation = normalizer.normalize_obs(batch)

    action, _ = model.predict(
        normalized_observation,
        deterministic=True,
    )

    return np.asarray(action, dtype=np.float32).reshape(4)


def raw_step(env, action: np.ndarray) -> dict[str, np.ndarray]:
    """Execute one simulation step without early task termination."""
    env.robot.set_action(np.asarray(action, dtype=np.float32))
    env.sim.step()
    return env._get_obs()


def parse_seed_list(seed_list: str | None) -> list[int] | None:
    """Parse comma-separated seed values."""
    if seed_list is None or not seed_list.strip():
        return None

    seeds = []

    for item in seed_list.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))

    if not seeds:
        raise ValueError("--seed-list was provided but no valid seeds were parsed")

    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export lift-focused PandaPickAndPlace teacher demonstrations "
            "for improving grasp/lift behavior."
        )
    )

    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-list", type=str, default=None)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum number of seeds to try when --seed-list is not used. "
            "Defaults to max(3 * episodes, episodes + 50)."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=TEACHER_MAX_STEPS)
    parser.add_argument(
        "--min-lift-height",
        type=float,
        default=DEFAULT_MIN_LIFT_HEIGHT,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "models/PandaPickAndPlace/"
            "chencliu_tqc_PandaPickAndPlace_v3"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/teacher_datasets/"
            "panda_pickandplace_lift_teacher_20"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero")

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than zero")

    if args.min_lift_height <= 0:
        raise ValueError("--min-lift-height must be greater than zero")

    explicit_seeds = parse_seed_list(args.seed_list)

    if explicit_seeds is not None and len(explicit_seeds) < args.episodes:
        raise ValueError(
            "--seed-list must contain at least as many seeds as --episodes"
        )

    max_attempts = (
        args.max_attempts
        if args.max_attempts is not None
        else max(args.episodes * 3, args.episodes + 50)
    )

    if explicit_seeds is None and max_attempts < args.episodes:
        raise ValueError(
            "--max-attempts must be greater than or equal to --episodes"
        )

    model_path = args.model_dir / "tqc-PandaPickAndPlace-v3.zip"
    vec_path = args.model_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    if not vec_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize file: {vec_path}")

    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.out_dir}. "
                "Use --overwrite."
            )
        shutil.rmtree(args.out_dir)

    frames_dir = args.out_dir / "frames"
    episodes_dir = args.out_dir / "episodes"
    staging_dir = args.out_dir / ".staging"

    frames_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    normalization_env = DummyVecEnv([make_normalization_env])
    normalizer = VecNormalize.load(str(vec_path), normalization_env)
    normalizer.training = False
    normalizer.norm_reward = False

    model = TQC.load(
        str(model_path),
        env=normalization_env,
        device="cpu",
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
        },
    )

    env = make_eval_env()

    csv_path = args.out_dir / "teacher_steps.csv"

    csv_fields = [
        "episode",
        "seed",
        "step",
        "phase",
        "task",
        "frame_path",
        "state_25",
        "env_action_4d",
        "smolvla_action_6d",
        "reward",
        "done",
        "is_success",
        "distance_to_goal",
    ]

    all_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []
    total_steps = 0
    attempted_seed_count = 0

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()

            episode = 0

            while episode < args.episodes:
                if explicit_seeds is not None:
                    if attempted_seed_count >= len(explicit_seeds):
                        break
                    seed = explicit_seeds[attempted_seed_count]
                else:
                    if attempted_seed_count >= max_attempts:
                        break
                    seed = args.seed_start + attempted_seed_count

                attempted_seed_count += 1

                observation, _ = env.reset(seed=seed)

                final_goal = np.asarray(
                    observation["desired_goal"],
                    dtype=np.float64,
                )
                final_goal[2] = OBJECT_Z
                env.task.goal = final_goal.copy()
                env.sim.set_base_pose(
                    "target",
                    final_goal,
                    np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                )
                observation = env._get_obs()

                hover_goal = final_goal.copy()
                hover_goal[2] = HOVER_Z

                initial_cube_z = float(observation["achieved_goal"][2])
                max_cube_z = initial_cube_z

                episode_frame_dir = staging_dir / f"seed_{seed:06d}"
                shutil.rmtree(episode_frame_dir, ignore_errors=True)
                episode_frame_dir.mkdir(parents=True, exist_ok=True)

                episode_rows = []
                step_index = 0
                lift_success = False
                best_goal_distance = float("inf")

                for _ in range(args.max_steps):
                    action = predict_teacher(
                        model,
                        normalizer,
                        observation,
                        hover_goal,
                    )

                    state_25 = build_state_25(observation)

                    cube_position = np.asarray(
                        observation["achieved_goal"],
                        dtype=np.float64,
                    )

                    distance_to_goal = float(
                        np.linalg.norm(cube_position - final_goal)
                    )
                    best_goal_distance = min(best_goal_distance, distance_to_goal)

                    frame_rel_path = (
                        Path("frames")
                        / f"episode_{episode:04d}"
                        / f"frame_{step_index:04d}.png"
                    )
                    frame_abs_path = (
                        episode_frame_dir / f"frame_{step_index:04d}.png"
                    )

                    save_rgb_frame(env.render(), frame_abs_path)

                    next_observation = raw_step(env, action)

                    next_cube_position = np.asarray(
                        next_observation["achieved_goal"],
                        dtype=np.float64,
                    )

                    max_cube_z = max(max_cube_z, float(next_cube_position[2]))
                    lift_height = max_cube_z - initial_cube_z

                    lift_success = lift_height >= args.min_lift_height

                    reward = 0.0 if lift_success else -1.0

                    row = {
                        "episode": episode,
                        "seed": seed,
                        "step": step_index,
                        "phase": "lift_teacher",
                        "task": TASK_TEXT,
                        "frame_path": str(frame_rel_path),
                        "state_25": json.dumps(state_25.tolist()),
                        "env_action_4d": json.dumps(action.tolist()),
                        "smolvla_action_6d": json.dumps(
                            action_4d_to_smolvla_6d(action).tolist()
                        ),
                        "reward": reward,
                        "done": lift_success,
                        "is_success": lift_success,
                        "distance_to_goal": distance_to_goal,
                    }

                    episode_rows.append(row)
                    observation = next_observation
                    step_index += 1

                    if lift_success:
                        break

                if not lift_success:
                    rejected_attempts.append(
                        {
                            "seed": seed,
                            "reason": "lift_failure",
                            "attempted_steps": len(episode_rows),
                            "max_lift_height": max_cube_z - initial_cube_z,
                            "best_goal_distance": best_goal_distance,
                        }
                    )

                    shutil.rmtree(episode_frame_dir, ignore_errors=True)

                    print(
                        f"REJECT seed={seed} | "
                        f"reason=lift_failure | "
                        f"max_lift={max_cube_z - initial_cube_z:.4f}"
                    )

                    continue

                final_episode_frame_dir = frames_dir / f"episode_{episode:04d}"

                if final_episode_frame_dir.exists():
                    raise FileExistsError(
                        "Accepted episode frame directory already exists: "
                        f"{final_episode_frame_dir}"
                    )

                episode_frame_dir.rename(final_episode_frame_dir)

                all_rows.extend(episode_rows)
                total_steps += len(episode_rows)

                final_cube_position = np.asarray(
                    observation["achieved_goal"],
                    dtype=np.float64,
                )

                final_distance = float(
                    np.linalg.norm(final_cube_position - final_goal)
                )

                episode_summary = {
                    "episode": episode,
                    "seed": seed,
                    "success": lift_success,
                    "steps": len(episode_rows),
                    "lift_height": max_cube_z - initial_cube_z,
                    "initial_cube_z": initial_cube_z,
                    "max_cube_z": max_cube_z,
                    "best_goal_distance": best_goal_distance,
                    "final_distance": final_distance,
                    "final_cube_position": final_cube_position.tolist(),
                }

                episode_summaries.append(episode_summary)

                write_json(
                    episodes_dir / f"episode_{episode:04d}.json",
                    {
                        "episode_summary": episode_summary,
                        "steps": episode_rows,
                    },
                )

                print(
                    f"EP {episode:04d} | "
                    f"seed={seed} | "
                    f"success={lift_success} | "
                    f"steps={len(episode_rows)} | "
                    f"lift={episode_summary['lift_height']:.4f} | "
                    f"best_goal_distance={best_goal_distance:.4f}"
                )

                episode += 1

            if episode < args.episodes:
                raise RuntimeError(
                    "Could not collect the requested number of successful "
                    "lift episodes."
                )

            writer.writerows(all_rows)

        metadata = {
            "env_id": "PandaPickAndPlace-v3",
            "dataset_variant": "lift_teacher",
            "teacher": "chencliu/tqc-PandaPickAndPlace-v3",
            "model_path": str(model_path),
            "vec_normalize_path": str(vec_path),
            "task": TASK_TEXT,
            "state_shape": [25],
            "env_action_shape": [4],
            "smolvla_action_shape": [6],
            "image_source": "env.render()",
            "image_format": "png",
            "requested_episodes": args.episodes,
            "episodes": len(episode_summaries),
            "seed_start": args.seed_start,
            "seed_list_used": explicit_seeds is not None,
            "max_attempts": max_attempts,
            "attempted_seed_count": attempted_seed_count,
            "accepted_seeds": [
                item["seed"] for item in episode_summaries
            ],
            "rejected_seed_count": len(rejected_attempts),
            "rejected_attempts": rejected_attempts,
            "export_complete": True,
            "total_steps": total_steps,
            "min_lift_height": args.min_lift_height,
            "max_steps": args.max_steps,
            "hover_z": HOVER_Z,
        }

        summary = {
            **metadata,
            "success_count": len(episode_summaries),
            "success_rate": 1.0,
            "mean_steps": float(
                np.mean([item["steps"] for item in episode_summaries])
            ),
            "mean_lift_height": float(
                np.mean([item["lift_height"] for item in episode_summaries])
            ),
            "mean_best_goal_distance": float(
                np.mean(
                    [item["best_goal_distance"] for item in episode_summaries]
                )
            ),
            "episode_summaries": episode_summaries,
        }

        write_json(args.out_dir / "metadata.json", metadata)
        write_json(args.out_dir / "summary.json", summary)

        print("\nEXPORT_OK")
        print(f"Output directory: {args.out_dir}")
        print(f"CSV: {csv_path}")
        print(f"Total frames: {total_steps}")
        print(f"Success: {summary['success_count']}/{args.episodes}")
        print(f"Attempted seeds: {attempted_seed_count}")
        print(f"Rejected seeds: {len(rejected_attempts)}")
        print(f"Mean steps: {summary['mean_steps']:.2f}")
        print(f"Mean lift height: {summary['mean_lift_height']:.4f}")

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        env.close()
        normalization_env.close()


if __name__ == "__main__":
    main()
