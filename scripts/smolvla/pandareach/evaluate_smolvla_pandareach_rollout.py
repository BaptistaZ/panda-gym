import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import panda_gym  # noqa: F401
except Exception:
    pass

import gymnasium as gym

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def to_tensor(value, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def resize_rgb_to_chw_float(image, size=256):
    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")

    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).contiguous()


def build_state(obs):
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)

    if achieved.shape[0] != 3 or desired.shape[0] != 3:
        raise ValueError(f"Expected achieved_goal and desired_goal with shape 3, got {achieved.shape}, {desired.shape}")

    return np.concatenate([achieved, desired], axis=0).astype(np.float32)


def distance(obs):
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))


def maybe_postprocess_action(postprocessor, action):
    if postprocessor is None:
        return action

    for candidate in [{"action": action}, action]:
        try:
            out = postprocessor(candidate)

            if isinstance(out, dict) and "action" in out:
                return out["action"]

            if isinstance(out, torch.Tensor):
                return out

        except Exception:
            pass

    return action


class SmolVLARolloutPolicy:
    def __init__(self, model_path, dataset_stats, device):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model_path).to(device).eval()
        self.policy.config.device = device

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            dataset_stats=dataset_stats,
        )

    def act(self, obs, rgb_image):
        state = build_state(obs)
        image = resize_rgb_to_chw_float(rgb_image)

        raw_batch = {
            "observation.state": to_tensor(state),
            "task": "reach the target",
        }

        for key in IMAGE_KEYS:
            raw_batch[key] = image.detach().clone()

        processed = self.preprocessor(raw_batch)

        for key, value in list(processed.items()):
            if isinstance(value, torch.Tensor):
                processed[key] = value.to(self.device)

        with torch.no_grad():
            raw_action = self.policy.select_action(processed)
            final_action = maybe_postprocess_action(self.postprocessor, raw_action)

        action = final_action.detach().cpu().numpy().reshape(-1)[:3].astype(np.float32)
        return action

    def close(self):
        del self.policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_env(render_mode="rgb_array"):
    return gym.make("PandaReach-v3", render_mode=render_mode)


def rollout_episode(policy_name, action_fn, seed, max_steps, action_scale=1.0):
    env = make_env()
    obs, info = env.reset(seed=seed)

    initial_distance = distance(obs)
    best_distance = initial_distance
    final_distance = initial_distance
    success_seen = False

    trajectory = []

    for step in range(max_steps):
        rgb = env.render()

        raw_action = np.asarray(action_fn(obs, rgb), dtype=np.float32).reshape(-1)
        scaled_action = raw_action[:3] * float(action_scale)
        action = np.clip(scaled_action, env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, info = env.step(action)

        current_distance = distance(next_obs)
        best_distance = min(best_distance, current_distance)
        final_distance = current_distance

        is_success = bool(info.get("is_success", False))
        success_seen = success_seen or is_success

        trajectory.append({
            "step": int(step),
            "action": action.tolist(),
            "raw_action": raw_action.tolist(),
            "reward": float(reward),
            "distance": current_distance,
            "is_success": is_success,
        })

        obs = next_obs

        if terminated or truncated or is_success:
            break

    env.close()

    return {
        "policy": policy_name,
        "seed": int(seed),
        "steps": len(trajectory),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "best_distance": best_distance,
        "distance_improvement": initial_distance - best_distance,
        "success_seen": success_seen,
        "trajectory": trajectory,
    }


def summarize(results):
    summary = {}

    for policy_name in sorted(set(row["policy"] for row in results)):
        rows = [row for row in results if row["policy"] == policy_name]

        summary[policy_name] = {
            "num_episodes": len(rows),
            "success_rate": float(np.mean([row["success_seen"] for row in rows])),
            "mean_initial_distance": float(np.mean([row["initial_distance"] for row in rows])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in rows])),
            "mean_best_distance": float(np.mean([row["best_distance"] for row in rows])),
            "mean_distance_improvement": float(np.mean([row["distance_improvement"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
        }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="outputs/lerobot_datasets/panda_reach_ppo_teacher_50")
    parser.add_argument("--repo-id", default="local/panda_reach_ppo_teacher_50")
    parser.add_argument("--base-model", default="lerobot/smolvla_base")
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/smolvla_panda_reach_chunk1_50/checkpoints/000050/pretrained_model",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="outputs/smolvla_rollout/pandareach_chunk1_50")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--smolvla-action-scale", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LeRobotDataset(args.repo_id, root=Path(args.dataset_root))
    dataset_stats = dataset.meta.stats

    seeds = [args.seed + i for i in range(args.episodes)]

    results = []

    for seed in seeds:
        results.append(
            rollout_episode(
                "zero",
                lambda obs, rgb: np.zeros(3, dtype=np.float32),
                seed,
                args.max_steps,
                1.0,
            )
        )

    for seed in seeds:
        rng = np.random.default_rng(seed)
        results.append(
            rollout_episode(
                "random",
                lambda obs, rgb, rng=rng: rng.uniform(-1.0, 1.0, size=3).astype(np.float32),
                seed,
                args.max_steps,
                1.0,
            )
        )

    if not args.skip_base:
        print("Loading base SmolVLA policy...")
        base_policy = SmolVLARolloutPolicy(args.base_model, dataset_stats, device)
        for seed in seeds:
            results.append(
                rollout_episode(
                    "smolvla_base",
                    base_policy.act,
                    seed,
                    args.max_steps,
                    args.smolvla_action_scale,
                )
            )
        base_policy.close()

    print("Loading trained SmolVLA checkpoint...")
    checkpoint_policy = SmolVLARolloutPolicy(args.checkpoint, dataset_stats, device)
    for seed in seeds:
        results.append(
            rollout_episode(
                "smolvla_checkpoint",
                checkpoint_policy.act,
                seed,
                args.max_steps,
                args.smolvla_action_scale,
            )
        )
    checkpoint_policy.close()

    summary = summarize(results)

    output = {
        "ok": True,
        "purpose": "Evaluate SmolVLA checkpoint in PandaReach-v3 rollout",
        "note": "This is an environment rollout test, not only offline action proximity.",
        "device": device,
        "repo_id": args.repo_id,
        "dataset_root": args.dataset_root,
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "smolvla_action_scale": args.smolvla_action_scale,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "summary": summary,
        "episodes_detail": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "rollout_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["summary"], indent=2))
    print(f"saved_result={out_path}")


if __name__ == "__main__":
    main()
