import argparse
import gc
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


def to_tensor(value, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def build_raw_batch(sample):
    batch = {
        "observation.state": to_tensor(sample["observation.state"]),
        "task": str(sample["task"]),
    }

    for key in IMAGE_KEYS:
        image = to_tensor(sample[key])
        if tuple(image.shape) != (3, 256, 256):
            raise ValueError(f"{key} has invalid shape {tuple(image.shape)}")
        batch[key] = image

    return batch


def maybe_postprocess_action(postprocessor, action):
    if postprocessor is None:
        return action, "no_postprocessor"

    errors = []

    for candidate in [{"action": action}, action]:
        try:
            out = postprocessor(candidate)

            if isinstance(out, dict) and "action" in out:
                return out["action"], "dict_action"

            if isinstance(out, torch.Tensor):
                return out, "tensor"

        except Exception as exc:
            errors.append(repr(exc))

    return action, f"postprocessor_failed: {errors}"


def infer_actions(model_id_or_path, samples, device, dataset_stats=None):
    policy = SmolVLAPolicy.from_pretrained(model_id_or_path).to(device).eval()
    policy.config.device = device

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_stats,
    )

    results = []

    with torch.no_grad():
        for sample in samples:
            raw_batch = build_raw_batch(sample)
            processed = preprocessor(raw_batch)

            for key, value in list(processed.items()):
                if isinstance(value, torch.Tensor):
                    processed[key] = value.to(device)

            raw_action = policy.select_action(processed)
            final_action, postprocess_status = maybe_postprocess_action(postprocessor, raw_action)

            action_np = final_action.detach().cpu().numpy().reshape(-1)

            results.append({
                "action": action_np,
                "postprocess_status": postprocess_status,
            })

    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def l2(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def make_sample_indices(num_frames, requested_indices, max_samples):
    if requested_indices:
        return requested_indices

    if max_samples >= num_frames:
        return list(range(num_frames))

    return np.linspace(0, num_frames - 1, num=max_samples, dtype=int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/lerobot_datasets/panda_reach_ppo_teacher_50")
    parser.add_argument("--repo-id", default="local/panda_reach_ppo_teacher_50")
    parser.add_argument("--base-model", default="lerobot/smolvla_base")
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/smolvla_panda_reach_checkpoint_test/checkpoints/000020/pretrained_model",
    )
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--out-dir", default="outputs/smolvla_probe/checkpoint_comparison")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LeRobotDataset(args.repo_id, root=Path(args.root))

    dataset_stats = None

    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats"):
        dataset_stats = dataset.meta.stats
    elif hasattr(dataset, "stats"):
        dataset_stats = dataset.stats

    sample_indices = make_sample_indices(dataset.num_frames, args.sample_indices, args.max_samples)

    samples = []
    teacher_actions = []

    for index in sample_indices:
        if index < 0 or index >= dataset.num_frames:
            raise IndexError(f"sample index {index} is outside range [0, {dataset.num_frames - 1}]")

        sample = dataset[index]
        samples.append(sample)
        teacher_actions.append(to_tensor(sample["action"]).detach().cpu().numpy().reshape(-1))

    print(f"Using {len(sample_indices)} samples:", sample_indices)

    print("Loading base model and inferring actions...")
    base_results = infer_actions(args.base_model, samples, device, dataset_stats=dataset_stats)

    print("Loading trained checkpoint and inferring actions...")
    checkpoint_results = infer_actions(args.checkpoint, samples, device, dataset_stats=dataset_stats)

    comparisons = []

    for i, sample_index in enumerate(sample_indices):
        teacher_action = teacher_actions[i]
        base_action = base_results[i]["action"]
        checkpoint_action = checkpoint_results[i]["action"]

        base_l2_all = l2(base_action, teacher_action)
        checkpoint_l2_all = l2(checkpoint_action, teacher_action)

        base_l2_real = l2(base_action[:3], teacher_action[:3])
        checkpoint_l2_real = l2(checkpoint_action[:3], teacher_action[:3])

        base_l2_padded = l2(base_action[3:6], teacher_action[3:6])
        checkpoint_l2_padded = l2(checkpoint_action[3:6], teacher_action[3:6])

        comparisons.append({
            "sample_index": int(sample_index),
            "episode_index": int(samples[i]["episode_index"].item()),
            "frame_index": int(samples[i]["frame_index"].item()),
            "task": str(samples[i]["task"]),
            "teacher_action_6d": teacher_action.tolist(),
            "base_action_6d": base_action.tolist(),
            "checkpoint_action_6d": checkpoint_action.tolist(),
            "base_to_teacher_l2_all_6d": base_l2_all,
            "checkpoint_to_teacher_l2_all_6d": checkpoint_l2_all,
            "l2_improvement_all_6d_positive_is_better": base_l2_all - checkpoint_l2_all,
            "base_to_teacher_l2_real_dims_0_2": base_l2_real,
            "checkpoint_to_teacher_l2_real_dims_0_2": checkpoint_l2_real,
            "l2_improvement_real_dims_0_2_positive_is_better": base_l2_real - checkpoint_l2_real,
            "base_to_teacher_l2_padded_dims_3_5": base_l2_padded,
            "checkpoint_to_teacher_l2_padded_dims_3_5": checkpoint_l2_padded,
            "l2_improvement_padded_dims_3_5_positive_is_better": base_l2_padded - checkpoint_l2_padded,
            "base_postprocess_status": base_results[i]["postprocess_status"],
            "checkpoint_postprocess_status": checkpoint_results[i]["postprocess_status"],
        })

    def mean(key):
        return float(np.mean([row[key] for row in comparisons]))

    result = {
        "ok": True,
        "purpose": "Compare SmolVLA base and trained checkpoint against PPO teacher actions",
        "note": (
            "This checks numerical action proximity on dataset samples only. "
            "It does not prove task success in PandaReach."
        ),
        "device": device,
        "repo_id": args.repo_id,
        "root": args.root,
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "num_frames": int(dataset.num_frames),
        "used_dataset_stats": dataset_stats is not None,
        "sample_indices": sample_indices,
        "num_samples": len(sample_indices),
        "mean_base_to_teacher_l2_all_6d": mean("base_to_teacher_l2_all_6d"),
        "mean_checkpoint_to_teacher_l2_all_6d": mean("checkpoint_to_teacher_l2_all_6d"),
        "mean_l2_improvement_all_6d_positive_is_better": mean("l2_improvement_all_6d_positive_is_better"),
        "mean_base_to_teacher_l2_real_dims_0_2": mean("base_to_teacher_l2_real_dims_0_2"),
        "mean_checkpoint_to_teacher_l2_real_dims_0_2": mean("checkpoint_to_teacher_l2_real_dims_0_2"),
        "mean_l2_improvement_real_dims_0_2_positive_is_better": mean("l2_improvement_real_dims_0_2_positive_is_better"),
        "mean_base_to_teacher_l2_padded_dims_3_5": mean("base_to_teacher_l2_padded_dims_3_5"),
        "mean_checkpoint_to_teacher_l2_padded_dims_3_5": mean("checkpoint_to_teacher_l2_padded_dims_3_5"),
        "mean_l2_improvement_padded_dims_3_5_positive_is_better": mean("l2_improvement_padded_dims_3_5_positive_is_better"),
        "comparisons": comparisons,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "comparison_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"saved_result={out_path}")


if __name__ == "__main__":
    main()
