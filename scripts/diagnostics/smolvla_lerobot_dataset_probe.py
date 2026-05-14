import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def tensor_summary(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min().item()) if value.numel() else None,
            "max": float(value.max().item()) if value.numel() else None,
            "mean": float(value.float().mean().item()) if value.numel() else None,
        }

    array = np.asarray(value)
    return {
        "type": type(value).__name__,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def prepare_tensor(value, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)

    return torch.tensor(value, dtype=dtype)


def build_raw_smolvla_batch(sample):
    state = prepare_tensor(sample["observation.state"], dtype=torch.float32)

    raw_batch = {
        "observation.state": state,
        "task": sample["task"],
    }

    for key in IMAGE_KEYS:
        image = prepare_tensor(sample[key], dtype=torch.float32)

        if image.shape != (3, 256, 256):
            raise ValueError(f"{key} has invalid shape {tuple(image.shape)}. Expected (3, 256, 256).")

        raw_batch[key] = image

    return raw_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/lerobot_datasets/panda_reach_ppo_teacher_50")
    parser.add_argument("--repo-id", default="local/panda_reach_ppo_teacher_50")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--out-dir", default="outputs/smolvla_probe/lerobot_dataset_probe")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LeRobotDataset(args.repo_id, root=Path(args.root))

    if args.sample_index < 0 or args.sample_index >= dataset.num_frames:
        raise IndexError(
            f"sample-index={args.sample_index} is outside dataset range [0, {dataset.num_frames - 1}]"
        )

    sample = dataset[args.sample_index]

    raw_batch = build_raw_smolvla_batch(sample)
    teacher_action = prepare_tensor(sample["action"], dtype=torch.float32)

    policy = SmolVLAPolicy.from_pretrained(args.model_id).to(device).eval()
    policy.config.device = device

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        dataset_stats=None,
    )

    processed = preprocessor(raw_batch)

    with torch.no_grad():
        smolvla_action = policy.select_action(processed)

    teacher_action_np = teacher_action.detach().cpu().numpy().reshape(-1)
    smolvla_action_np = smolvla_action.detach().cpu().numpy().reshape(-1)

    action_delta = smolvla_action_np - teacher_action_np
    action_delta_l2 = float(np.linalg.norm(action_delta))

    result = {
        "ok": True,
        "purpose": "LeRobotDataset sample to SmolVLA runtime compatibility probe",
        "note": (
            "This is not fine-tuning and does not evaluate task success. "
            "The SmolVLA base action is not expected to match the PPO teacher action."
        ),
        "device": device,
        "repo_id": args.repo_id,
        "root": args.root,
        "num_episodes": int(dataset.num_episodes),
        "num_frames": int(dataset.num_frames),
        "sample_index": int(args.sample_index),
        "sample_task": sample["task"],
        "sample_episode_index": int(sample["episode_index"].item()),
        "sample_frame_index": int(sample["frame_index"].item()),
        "sample_timestamp": float(sample["timestamp"].item()),
        "sample_summaries": {
            "observation.state": tensor_summary(sample["observation.state"]),
            "observation.images.camera1": tensor_summary(sample["observation.images.camera1"]),
            "observation.images.camera2": tensor_summary(sample["observation.images.camera2"]),
            "observation.images.camera3": tensor_summary(sample["observation.images.camera3"]),
            "teacher_action": tensor_summary(teacher_action),
        },
        "processed_keys": sorted(list(processed.keys())),
        "processed_summaries": {
            key: tensor_summary(value)
            for key, value in processed.items()
            if isinstance(value, torch.Tensor)
        },
        "teacher_action_6d": teacher_action_np.tolist(),
        "smolvla_action_6d": smolvla_action_np.tolist(),
        "action_delta_6d": action_delta.tolist(),
        "action_delta_l2": action_delta_l2,
    }

    result_path = out_dir / f"sample_{args.sample_index:06d}_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"saved_result={result_path}")


if __name__ == "__main__":
    main()
