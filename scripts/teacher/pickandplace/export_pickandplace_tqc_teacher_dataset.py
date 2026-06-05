from pathlib import Path
import argparse
import csv
import json
import shutil

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
from PIL import Image

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


TASK_TEXT = "pick and place the cube"


def make_env():
    return gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")


def build_state_25(raw_obs: dict) -> np.ndarray:
    observation = np.asarray(raw_obs["observation"][0], dtype=np.float32)
    achieved_goal = np.asarray(raw_obs["achieved_goal"][0], dtype=np.float32)
    desired_goal = np.asarray(raw_obs["desired_goal"][0], dtype=np.float32)

    state = np.concatenate([observation, achieved_goal, desired_goal]).astype(np.float32)

    if state.shape != (25,):
        raise ValueError(f"Expected state shape (25,), got {state.shape}")

    return state


def goal_distance(raw_obs: dict) -> float:
    achieved = np.asarray(raw_obs["achieved_goal"][0], dtype=np.float32)
    desired = np.asarray(raw_obs["desired_goal"][0], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))


def action_4d_to_smolvla_6d(action_4d: np.ndarray) -> np.ndarray:
    action_4d = np.asarray(action_4d, dtype=np.float32).reshape(4)
    action_6d = np.zeros(6, dtype=np.float32)
    action_6d[:4] = action_4d
    return action_6d


def save_rgb_frame(frame: np.ndarray, path: Path) -> None:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    Image.fromarray(frame).save(path)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/PandaPickAndPlace/chencliu_tqc_PandaPickAndPlace_v3"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/teacher_datasets/panda_pickandplace_tqc_10"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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
                "Use --overwrite to replace it."
            )
        shutil.rmtree(args.out_dir)

    frames_dir = args.out_dir / "frames"
    episodes_dir = args.out_dir / "episodes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(str(vec_path), venv)
    venv.training = False
    venv.norm_reward = False

    model = TQC.load(
        str(model_path),
        env=venv,
        device="cpu",
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
        },
    )

    csv_path = args.out_dir / "teacher_steps.csv"

    csv_fields = [
        "episode",
        "step",
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

    all_episode_summaries = []
    total_steps = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for ep in range(args.episodes):
            obs = venv.reset()
            raw_obs = venv.get_original_obs()

            episode_frame_dir = frames_dir / f"episode_{ep:04d}"
            episode_frame_dir.mkdir(parents=True, exist_ok=True)

            episode_rows = []
            episode_success = False
            episode_reward = 0.0
            episode_best_distance = goal_distance(raw_obs)
            episode_initial_distance = episode_best_distance

            for step in range(args.max_steps):
                raw_obs = venv.get_original_obs()
                state_25 = build_state_25(raw_obs)
                distance = goal_distance(raw_obs)

                frame = venv.envs[0].render()
                frame_rel_path = Path("frames") / f"episode_{ep:04d}" / f"frame_{step:04d}.png"
                frame_abs_path = args.out_dir / frame_rel_path
                save_rgb_frame(frame, frame_abs_path)

                action, _ = model.predict(obs, deterministic=True)
                env_action_4d = np.asarray(action[0], dtype=np.float32)
                smolvla_action_6d = action_4d_to_smolvla_6d(env_action_4d)

                obs, reward, done, info = venv.step(action)

                reward_value = float(reward[0])
                done_value = bool(done[0])
                success_value = bool(info[0].get("is_success", False))

                episode_reward += reward_value
                episode_best_distance = min(episode_best_distance, distance)

                if success_value:
                    episode_success = True

                row = {
                    "episode": ep,
                    "step": step,
                    "task": TASK_TEXT,
                    "frame_path": str(frame_rel_path),
                    "state_25": json.dumps(state_25.tolist()),
                    "env_action_4d": json.dumps(env_action_4d.tolist()),
                    "smolvla_action_6d": json.dumps(smolvla_action_6d.tolist()),
                    "reward": reward_value,
                    "done": done_value,
                    "is_success": success_value,
                    "distance_to_goal": distance,
                }

                writer.writerow(row)
                episode_rows.append(row)
                total_steps += 1

                if done_value:
                    break

            episode_summary = {
                "episode": ep,
                "success": episode_success,
                "steps": len(episode_rows),
                "reward": episode_reward,
                "initial_distance": episode_initial_distance,
                "best_distance": episode_best_distance,
                "final_distance": episode_rows[-1]["distance_to_goal"] if episode_rows else None,
            }

            all_episode_summaries.append(episode_summary)

            write_json(
                episodes_dir / f"episode_{ep:04d}.json",
                {
                    "episode_summary": episode_summary,
                    "steps": episode_rows,
                },
            )

            print(
                f"EP {ep:04d} | "
                f"success={episode_success} | "
                f"steps={len(episode_rows)} | "
                f"reward={episode_reward:.2f} | "
                f"best_distance={episode_best_distance:.4f}"
            )

    success_values = [episode["success"] for episode in all_episode_summaries]

    metadata = {
        "env_id": "PandaPickAndPlace-v3",
        "teacher": "chencliu/tqc-PandaPickAndPlace-v3",
        "model_path": str(model_path),
        "vec_normalize_path": str(vec_path),
        "task": TASK_TEXT,
        "state_shape": [25],
        "env_action_shape": [4],
        "smolvla_action_shape": [6],
        "image_source": "env.render()",
        "image_format": "png",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "total_steps": total_steps,
    }

    summary = {
        **metadata,
        "success_rate": float(np.mean(success_values)),
        "success_count": int(np.sum(success_values)),
        "mean_steps": float(np.mean([e["steps"] for e in all_episode_summaries])),
        "mean_reward": float(np.mean([e["reward"] for e in all_episode_summaries])),
        "mean_initial_distance": float(np.mean([e["initial_distance"] for e in all_episode_summaries])),
        "mean_best_distance": float(np.mean([e["best_distance"] for e in all_episode_summaries])),
        "mean_final_distance": float(np.mean([e["final_distance"] for e in all_episode_summaries])),
        "episode_summaries": all_episode_summaries,
    }

    write_json(args.out_dir / "metadata.json", metadata)
    write_json(args.out_dir / "summary.json", summary)

    print("\nEXPORT_OK")
    print(f"Output directory: {args.out_dir}")
    print(f"CSV: {csv_path}")
    print(f"Total steps: {total_steps}")
    print(f"Success rate: {summary['success_rate']:.4f}")
    print(f"Success count: {summary['success_count']}/{args.episodes}")

    venv.close()


if __name__ == "__main__":
    main()
