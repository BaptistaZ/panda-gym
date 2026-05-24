#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def resize_rgb_to_chw_float(image: np.ndarray, size: int = 256) -> torch.Tensor:
    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")

    arr = np.ascontiguousarray(arr)
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).contiguous()


def build_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    observation = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)

    if observation.shape[0] != 18:
        raise ValueError(f"Expected observation with shape 18, got {observation.shape}")

    if achieved.shape[0] != 3 or desired.shape[0] != 3:
        raise ValueError(f"Expected achieved_goal and desired_goal with shape 3, got {achieved.shape}, {desired.shape}")

    return np.concatenate([observation, achieved, desired], axis=0).astype(np.float32)


def distance(obs: dict[str, np.ndarray]) -> float:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))


def maybe_postprocess_action(postprocessor: Any, action: torch.Tensor) -> torch.Tensor:
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


class SmolVLAPandaPushPolicy:
    def __init__(self, checkpoint: str | Path, dataset_stats: dict, device: str) -> None:
        self.device = device
        self.checkpoint = str(checkpoint)
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint).to(device).eval()
        self.policy.config.device = device

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            dataset_stats=dataset_stats,
        )

    def act(
        self,
        obs: dict[str, np.ndarray],
        rgb_image: np.ndarray,
        task: str = "push the cube to the target",
    ) -> np.ndarray:
        state = build_state(obs)
        image = resize_rgb_to_chw_float(rgb_image)

        raw_batch: dict[str, Any] = {
            "observation.state": to_tensor(state),
            "task": task,
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

        action_6d = final_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        action_3d = action_6d[:3]
        return action_3d

    def close(self) -> None:
        del self.policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_env(env_id: str, render_mode: str = "rgb_array") -> gym.Env:
    return gym.make(env_id, render_mode=render_mode)


def rollout_episode(
    env_id: str,
    policy_name: str,
    action_fn,
    seed: int,
    max_steps: int,
    action_scale: float,
) -> dict[str, Any]:
    env = make_env(env_id)
    obs, info = env.reset(seed=seed)

    initial_distance = distance(obs)
    best_distance = initial_distance
    final_distance = initial_distance
    success_seen = bool(info.get("is_success", False))

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

        trajectory.append(
            {
                "step": int(step),
                "action": action.tolist(),
                "raw_action": raw_action.tolist(),
                "reward": float(reward),
                "distance": current_distance,
                "is_success": is_success,
            }
        )

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


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PandaPush-v3")
    parser.add_argument("--dataset-root", default="outputs/lerobot_datasets/panda_push_sac_her_teacher_100")
    parser.add_argument("--repo-id", default="local/panda_push_sac_her_teacher_100")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="outputs/smolvla_rollout/pandapush")
    parser.add_argument("--smolvla-action-scale", type=float, default=1.0)
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LeRobotDataset(args.repo_id, root=Path(args.dataset_root))
    dataset_stats = dataset.meta.stats

    seeds = [args.seed + i for i in range(args.episodes)]
    results: list[dict[str, Any]] = []

    if not args.skip_baselines:
        for seed in seeds:
            results.append(
                rollout_episode(
                    args.env,
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
                    args.env,
                    "random",
                    lambda obs, rgb, rng=rng: rng.uniform(-1.0, 1.0, size=3).astype(np.float32),
                    seed,
                    args.max_steps,
                    1.0,
                )
            )

    print("Loading SmolVLA checkpoint...")
    smolvla_policy = SmolVLAPandaPushPolicy(args.checkpoint, dataset_stats, device)

    try:
        for seed in seeds:
            results.append(
                rollout_episode(
                    args.env,
                    "smolvla_checkpoint",
                    lambda obs, rgb: smolvla_policy.act(obs, rgb, "push the cube to the target"),
                    seed,
                    args.max_steps,
                    args.smolvla_action_scale,
                )
            )
    finally:
        smolvla_policy.close()

    summary = summarize(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "env": args.env,
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "repo_id": args.repo_id,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "smolvla_action_scale": args.smolvla_action_scale,
        "summary": summary,
        "results": results,
    }

    (out_dir / "rollout_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
