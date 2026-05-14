import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def stats_for_array(array):
    array = np.asarray(array, dtype=np.float32)

    return {
        "shape": list(array.shape),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/lerobot_datasets/panda_reach_ppo_teacher_50")
    parser.add_argument("--repo-id", default="local/panda_reach_ppo_teacher_50")
    parser.add_argument("--out-dir", default="outputs/dataset_inspection/panda_reach_ppo_teacher_50")
    parser.add_argument("--max-camera-checks", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.repo_id, root=root)

    states = []
    actions = []
    tasks = []
    episode_lengths = defaultdict(int)
    frame_indices_by_episode = defaultdict(list)

    camera_equal_checks = {
        "camera1_eq_camera2": [],
        "camera1_eq_camera3": [],
        "camera2_eq_camera3": [],
    }

    for idx in range(dataset.num_frames):
        sample = dataset[idx]

        state = to_numpy(sample["observation.state"]).astype(np.float32).reshape(-1)
        action = to_numpy(sample["action"]).astype(np.float32).reshape(-1)

        states.append(state)
        actions.append(action)
        tasks.append(str(sample["task"]))

        episode_index = int(to_numpy(sample["episode_index"]).item())
        frame_index = int(to_numpy(sample["frame_index"]).item())

        episode_lengths[episode_index] += 1
        frame_indices_by_episode[episode_index].append(frame_index)

        if idx < args.max_camera_checks:
            cam1 = to_numpy(sample["observation.images.camera1"])
            cam2 = to_numpy(sample["observation.images.camera2"])
            cam3 = to_numpy(sample["observation.images.camera3"])

            camera_equal_checks["camera1_eq_camera2"].append(bool(np.allclose(cam1, cam2)))
            camera_equal_checks["camera1_eq_camera3"].append(bool(np.allclose(cam1, cam3)))
            camera_equal_checks["camera2_eq_camera3"].append(bool(np.allclose(cam2, cam3)))

    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)

    action_norms = np.linalg.norm(actions, axis=1)
    state_norms = np.linalg.norm(states, axis=1)

    padded_action_tail = actions[:, 3:6]
    padded_tail_all_zero = bool(np.allclose(padded_action_tail, 0.0))

    saturated_first_3 = np.isclose(np.abs(actions[:, :3]), 1.0, atol=1e-6)
    saturated_fraction_first_3 = saturated_first_3.mean(axis=0)

    frame_index_sequences_ok = {}
    for episode, frame_indices in frame_indices_by_episode.items():
        expected = list(range(len(frame_indices)))
        frame_index_sequences_ok[int(episode)] = frame_indices == expected

    result = {
        "ok": True,
        "purpose": "Inspect local LeRobot PPO teacher dataset statistics",
        "repo_id": args.repo_id,
        "root": str(root),
        "num_episodes": int(dataset.num_episodes),
        "num_frames": int(dataset.num_frames),
        "features": list(dataset.features.keys()),
        "task_counts": dict(Counter(tasks)),
        "episode_lengths": {
            str(k): int(v)
            for k, v in sorted(episode_lengths.items())
        },
        "episode_length_summary": {
            "min": int(min(episode_lengths.values())),
            "max": int(max(episode_lengths.values())),
            "mean": float(np.mean(list(episode_lengths.values()))),
        },
        "frame_index_sequences_all_ok": bool(all(frame_index_sequences_ok.values())),
        "state_stats": stats_for_array(states),
        "action_stats": stats_for_array(actions),
        "state_norm": {
            "min": float(state_norms.min()),
            "max": float(state_norms.max()),
            "mean": float(state_norms.mean()),
            "std": float(state_norms.std()),
        },
        "action_norm": {
            "min": float(action_norms.min()),
            "max": float(action_norms.max()),
            "mean": float(action_norms.mean()),
            "std": float(action_norms.std()),
        },
        "padded_action_dims_3_to_5_all_zero": padded_tail_all_zero,
        "saturated_fraction_first_3_action_dims": saturated_fraction_first_3.tolist(),
        "camera_replication_check": {
            key: {
                "checked_frames": len(values),
                "all_equal": bool(all(values)),
            }
            for key, values in camera_equal_checks.items()
        },
        "interpretation": {
            "padded_action_tail": (
                "Expected true for current PandaReach setup because PPO action is 3D "
                "and is padded to SmolVLA-compatible 6D action."
            ),
            "camera_replication": (
                "Expected true in this phase because one PandaGym render is replicated "
                "across camera1, camera2 and camera3 for SmolVLA format compatibility."
            ),
        },
    }

    result_path = out_dir / "inspection_summary.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"saved_result={result_path}")


if __name__ == "__main__":
    main()
