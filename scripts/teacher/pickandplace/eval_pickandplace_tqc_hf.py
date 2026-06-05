from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import gymnasium as gym
import panda_gym  # noqa: F401

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env():
    return gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")


def goal_distance_from_obs(raw_obs: dict) -> float:
    achieved = np.asarray(raw_obs["achieved_goal"][0], dtype=np.float32)
    desired = np.asarray(raw_obs["desired_goal"][0], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))


def as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/PandaPickAndPlace/chencliu_tqc_PandaPickAndPlace_v3"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evidence/PandaPickAndPlace/teacher_tqc_hf"),
    )
    args = parser.parse_args()

    model_path = args.model_dir / "tqc-PandaPickAndPlace-v3.zip"
    vec_path = args.model_dir / "vec_normalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    if not vec_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize file: {vec_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    episode_results = []

    for ep in range(args.episodes):
        obs = venv.reset()
        raw_obs = venv.get_original_obs()

        initial_distance = goal_distance_from_obs(raw_obs)
        best_distance = initial_distance
        last_valid_distance = initial_distance

        total_reward = 0.0
        success = False
        steps = 0
        done_reason = "max_steps"

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = venv.step(action)

            info0 = info[0]
            reward0 = float(reward[0])
            done0 = bool(done[0])

            total_reward += reward0
            steps = step + 1

            # Read distance only while the VecEnv has not auto-reset.
            if not done0:
                raw_obs = venv.get_original_obs()
                current_distance = goal_distance_from_obs(raw_obs)
                last_valid_distance = current_distance
                best_distance = min(best_distance, current_distance)

            if bool(info0.get("is_success", False)):
                success = True

            if done0:
                if bool(info0.get("is_success", False)):
                    done_reason = "success"
                elif bool(info0.get("TimeLimit.truncated", False)):
                    done_reason = "time_limit"
                else:
                    done_reason = "terminated"
                break

        final_distance = last_valid_distance

        result = {
            "episode": ep,
            "success": success,
            "initial_distance": initial_distance,
            "best_distance": best_distance,
            "final_distance": final_distance,
            "distance_improvement": initial_distance - best_distance,
            "steps": steps,
            "reward": total_reward,
            "done_reason": done_reason,
        }
        episode_results.append(result)

        print(
            f"EP {ep:03d} | "
            f"success={success} | "
            f"initial={initial_distance:.4f} | "
            f"best={best_distance:.4f} | "
            f"final={final_distance:.4f} | "
            f"steps={steps} | "
            f"reward={total_reward:.2f} | "
            f"done_reason={done_reason}"
        )

    success_values = [r["success"] for r in episode_results]

    summary = {
        "env_id": "PandaPickAndPlace-v3",
        "teacher": "chencliu/tqc-PandaPickAndPlace-v3",
        "model_path": str(model_path),
        "vec_normalize_path": str(vec_path),
        "n_episodes": args.episodes,
        "max_steps": args.max_steps,
        "success_rate": float(np.mean(success_values)),
        "success_count": int(np.sum(success_values)),
        "mean_initial_distance": float(np.mean([r["initial_distance"] for r in episode_results])),
        "mean_best_distance": float(np.mean([r["best_distance"] for r in episode_results])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in episode_results])),
        "mean_distance_improvement": float(np.mean([r["distance_improvement"] for r in episode_results])),
        "mean_steps": float(np.mean([r["steps"] for r in episode_results])),
        "mean_reward": float(np.mean([r["reward"] for r in episode_results])),
        "episodes": episode_results,
    }

    out_path = args.out_dir / f"eval_{args.episodes}_episodes_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme_path = args.out_dir / "README.md"
    readme_path.write_text(
        "# PandaPickAndPlace-v3 TQC Teacher Evaluation\n\n"
        "Teacher: `chencliu/tqc-PandaPickAndPlace-v3`\n\n"
        "Validated locally with `panda-smolvla`.\n\n"
        f"- Episodes: {args.episodes}\n"
        f"- Success rate: {summary['success_rate']:.4f}\n"
        f"- Success count: {summary['success_count']}/{args.episodes}\n"
        f"- Mean reward: {summary['mean_reward']:.4f}\n"
        f"- Mean steps: {summary['mean_steps']:.4f}\n"
        f"- Mean initial distance: {summary['mean_initial_distance']:.4f}\n"
        f"- Mean best distance: {summary['mean_best_distance']:.4f}\n"
        f"- Mean final distance: {summary['mean_final_distance']:.4f}\n",
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {out_path}")
    print(f"Saved README to: {readme_path}")

    venv.close()


if __name__ == "__main__":
    main()
