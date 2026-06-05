import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


TASK_TEXT = "pick and place the cube"

IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def build_state_25(obs: dict) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(obs["observation"], dtype=np.float32),
            np.asarray(obs["achieved_goal"], dtype=np.float32),
            np.asarray(obs["desired_goal"], dtype=np.float32),
        ]
    ).astype(np.float32)

    if state.shape != (25,):
        raise ValueError(f"Expected state shape (25,), got {state.shape}")

    return state


def goal_distance(obs: dict) -> float:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))


def render_to_chw_float(frame: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((256, 256), resample=Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    return torch.tensor(arr, dtype=torch.float32)


def maybe_postprocess_action(postprocessor, action: torch.Tensor) -> torch.Tensor:
    if postprocessor is None:
        return action

    for candidate in [{"action": action}, action]:
        try:
            output = postprocessor(candidate)

            if isinstance(output, dict) and "action" in output:
                return output["action"]

            if isinstance(output, torch.Tensor):
                return output

        except Exception:
            pass

    return action


def select_smolvla_action(
    policy,
    preprocessor,
    postprocessor,
    obs: dict,
    frame: np.ndarray,
    device: str,
) -> np.ndarray:
    state = torch.tensor(build_state_25(obs), dtype=torch.float32)
    image = render_to_chw_float(frame)

    raw_batch = {
        "observation.state": state,
        "task": TASK_TEXT,
    }

    # PandaGym provides one RGB render. It is replicated across the three
    # expected SmolVLA camera inputs for compatibility with the training dataset.
    for image_key in IMAGE_KEYS:
        raw_batch[image_key] = image

    processed = preprocessor(raw_batch)

    for key, value in list(processed.items()):
        if isinstance(value, torch.Tensor):
            processed[key] = value.to(device)

    with torch.no_grad():
        raw_action = policy.select_action(processed)
        action = maybe_postprocess_action(postprocessor, raw_action)

    action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)

    if action_np.shape != (6,):
        raise ValueError(f"Expected SmolVLA action shape (6,), got {action_np.shape}")

    return action_np


def get_dataset_stats(repo_id: str, root: Path):
    dataset = LeRobotDataset(repo_id, root=root)

    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats"):
        return dataset.meta.stats

    if hasattr(dataset, "stats"):
        return dataset.stats

    return None


def reset_policy_if_supported(policy) -> None:
    # Some LeRobot policies keep an internal action queue/cache.
    # Resetting it at episode boundaries avoids leaking state across rollouts.
    if hasattr(policy, "reset"):
        policy.reset()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/train/smolvla_panda_pickandplace_tqc_100_500/"
            "checkpoints/000500/pretrained_model"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/lerobot_datasets/panda_pickandplace_tqc_teacher_100"),
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="local/panda_pickandplace_tqc_teacher_100",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/smolvla_rollout/panda_pickandplace_500_rollout_5"),
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.action_scale <= 0:
        raise ValueError(f"Expected positive --action-scale, got {args.action_scale}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_stats = get_dataset_stats(args.dataset_repo_id, args.dataset_root)

    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to(device).eval()
    policy.config.device = device

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_stats,
    )

    env = gym.make("PandaPickAndPlace-v3", render_mode="rgb_array")

    episode_results = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            reset_policy_if_supported(policy)

            initial_distance = goal_distance(obs)
            best_distance = initial_distance
            final_distance = initial_distance
            total_reward = 0.0
            success = False
            steps = 0
            done_reason = "max_steps"

            step_rows = []

            for step in range(args.max_steps):
                frame = env.render()

                smolvla_action_6d = select_smolvla_action(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    obs=obs,
                    frame=frame,
                    device=device,
                )

                env_action_4d_raw = smolvla_action_6d[:4] * args.action_scale
                env_action_4d = np.clip(
                    env_action_4d_raw,
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)

                next_obs, reward, terminated, truncated, info = env.step(env_action_4d)

                distance = goal_distance(next_obs)

                total_reward += float(reward)
                best_distance = min(best_distance, distance)
                final_distance = distance
                steps = step + 1

                is_success = bool(info.get("is_success", False))

                if is_success:
                    success = True

                step_rows.append(
                    {
                        "step": step,
                        "distance": distance,
                        "reward": float(reward),
                        "is_success": is_success,
                        "smolvla_action_6d": smolvla_action_6d.tolist(),
                        "env_action_4d_raw": env_action_4d_raw.tolist(),
                        "env_action_4d": env_action_4d.tolist(),
                    }
                )

                obs = next_obs

                if terminated or truncated:
                    if is_success:
                        done_reason = "success"
                    elif truncated:
                        done_reason = "time_limit"
                    else:
                        done_reason = "terminated"
                    break

            result = {
                "episode": ep,
                "success": success,
                "steps": steps,
                "reward": total_reward,
                "initial_distance": initial_distance,
                "best_distance": best_distance,
                "final_distance": final_distance,
                "distance_improvement": initial_distance - best_distance,
                "done_reason": done_reason,
            }

            episode_results.append(result)

            episode_path = args.out_dir / f"episode_{ep:04d}.json"
            episode_path.write_text(
                json.dumps(
                    {
                        "summary": result,
                        "steps": step_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"EP {ep:04d} | "
                f"success={success} | "
                f"steps={steps} | "
                f"reward={total_reward:.2f} | "
                f"initial={initial_distance:.4f} | "
                f"best={best_distance:.4f} | "
                f"final={final_distance:.4f} | "
                f"done_reason={done_reason}"
            )

    finally:
        env.close()

    success_values = [r["success"] for r in episode_results]

    summary = {
        "env_id": "PandaPickAndPlace-v3",
        "checkpoint": str(args.checkpoint),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "device": device,
        "action_scale": args.action_scale,
        "success_rate": float(np.mean(success_values)),
        "success_count": int(np.sum(success_values)),
        "mean_steps": float(np.mean([r["steps"] for r in episode_results])),
        "mean_reward": float(np.mean([r["reward"] for r in episode_results])),
        "mean_initial_distance": float(np.mean([r["initial_distance"] for r in episode_results])),
        "mean_best_distance": float(np.mean([r["best_distance"] for r in episode_results])),
        "mean_final_distance": float(np.mean([r["final_distance"] for r in episode_results])),
        "mean_distance_improvement": float(
            np.mean([r["distance_improvement"] for r in episode_results])
        ),
        "episode_results": episode_results,
    }

    summary_path = args.out_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nROLLOUT_SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved_summary={summary_path}")


if __name__ == "__main__":
    main()