import argparse
import csv
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
            output = postprocessor(candidate)

            if isinstance(output, dict) and "action" in output:
                return output["action"], "dict_action"

            if isinstance(output, torch.Tensor):
                return output, "tensor"

        except Exception as exc:
            errors.append(repr(exc))

    return action, f"postprocessor_failed: {errors}"


def infer_actions(model_id_or_path, samples, device, dataset_stats=None, label="model"):
    policy = SmolVLAPolicy.from_pretrained(model_id_or_path).to(device).eval()
    policy.config.device = device

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_stats,
    )

    results = []

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            if idx % 25 == 0:
                print(f"{label}: inference {idx}/{len(samples)}")

            raw_batch = build_raw_batch(sample)
            processed = preprocessor(raw_batch)

            for key, value in list(processed.items()):
                if isinstance(value, torch.Tensor):
                    processed[key] = value.to(device)

            raw_action = policy.select_action(processed)
            final_action, postprocess_status = maybe_postprocess_action(postprocessor, raw_action)

            action_np = final_action.detach().cpu().numpy().reshape(-1).astype(np.float32)

            if action_np.shape != (6,):
                raise ValueError(f"{label} produced invalid action shape {action_np.shape}. Expected (6,).")

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


def abs_error(a, b):
    return np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))


def make_sample_indices(num_frames, requested_indices, max_samples):
    if requested_indices:
        return requested_indices

    if max_samples >= num_frames:
        return list(range(num_frames))

    return np.linspace(0, num_frames - 1, num=max_samples, dtype=int).tolist()


def summarize_values(values):
    values = np.asarray(values, dtype=np.float32)

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_per_dim(errors):
    errors = np.asarray(errors, dtype=np.float32)

    return {
        f"dim_{idx}": {
            "mean_abs_error": float(np.mean(errors[:, idx])),
            "median_abs_error": float(np.median(errors[:, idx])),
            "max_abs_error": float(np.max(errors[:, idx])),
        }
        for idx in range(errors.shape[1])
    }


def write_csv(path, rows):
    fieldnames = [
        "sample_index",
        "episode_index",
        "frame_index",
        "base_l2_6d",
        "checkpoint_l2_6d",
        "improvement_l2_6d",
        "base_l2_real_4d",
        "checkpoint_l2_real_4d",
        "improvement_l2_real_4d",
        "base_l2_padding_2d",
        "checkpoint_l2_padding_2d",
        "improvement_l2_padding_2d",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/lerobot_datasets/panda_pickandplace_tqc_teacher_100",
    )
    parser.add_argument(
        "--repo-id",
        default="local/panda_pickandplace_tqc_teacher_100",
    )
    parser.add_argument("--base-model", default="lerobot/smolvla_base")
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/smolvla_panda_pickandplace_tqc_100_500/checkpoints/000500/pretrained_model",
    )
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        default="outputs/smolvla_eval/panda_pickandplace_500_offline_compare",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LeRobotDataset(args.repo_id, root=Path(args.root))

    dataset_stats = None
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "stats"):
        dataset_stats = dataset.meta.stats
    elif hasattr(dataset, "stats"):
        dataset_stats = dataset.stats

    sample_indices = make_sample_indices(
        num_frames=dataset.num_frames,
        requested_indices=args.sample_indices,
        max_samples=args.max_samples,
    )

    samples = []
    teacher_actions = []

    for index in sample_indices:
        if index < 0 or index >= dataset.num_frames:
            raise IndexError(f"sample index {index} is outside range [0, {dataset.num_frames - 1}]")

        sample = dataset[index]
        samples.append(sample)

        teacher_action = to_tensor(sample["action"]).detach().cpu().numpy().reshape(-1).astype(np.float32)

        if teacher_action.shape != (6,):
            raise ValueError(f"Teacher action has invalid shape {teacher_action.shape}. Expected (6,).")

        teacher_actions.append(teacher_action)

    print(f"Using {len(sample_indices)} samples")
    print(f"Device: {device}")
    print("Loading base model and inferring actions...")
    base_results = infer_actions(
        args.base_model,
        samples,
        device,
        dataset_stats=dataset_stats,
        label="base",
    )

    print("Loading trained checkpoint and inferring actions...")
    checkpoint_results = infer_actions(
        args.checkpoint,
        samples,
        device,
        dataset_stats=dataset_stats,
        label="checkpoint",
    )

    comparisons = []

    base_l2_6d_values = []
    checkpoint_l2_6d_values = []
    base_l2_real_4d_values = []
    checkpoint_l2_real_4d_values = []
    base_l2_padding_2d_values = []
    checkpoint_l2_padding_2d_values = []

    base_abs_errors = []
    checkpoint_abs_errors = []

    for i, sample_index in enumerate(sample_indices):
        sample = samples[i]
        teacher_action = teacher_actions[i]
        base_action = base_results[i]["action"]
        checkpoint_action = checkpoint_results[i]["action"]

        base_l2_6d = l2(base_action, teacher_action)
        checkpoint_l2_6d = l2(checkpoint_action, teacher_action)

        base_l2_real_4d = l2(base_action[:4], teacher_action[:4])
        checkpoint_l2_real_4d = l2(checkpoint_action[:4], teacher_action[:4])

        base_l2_padding_2d = l2(base_action[4:6], teacher_action[4:6])
        checkpoint_l2_padding_2d = l2(checkpoint_action[4:6], teacher_action[4:6])

        base_l2_6d_values.append(base_l2_6d)
        checkpoint_l2_6d_values.append(checkpoint_l2_6d)
        base_l2_real_4d_values.append(base_l2_real_4d)
        checkpoint_l2_real_4d_values.append(checkpoint_l2_real_4d)
        base_l2_padding_2d_values.append(base_l2_padding_2d)
        checkpoint_l2_padding_2d_values.append(checkpoint_l2_padding_2d)

        base_abs_errors.append(abs_error(base_action, teacher_action))
        checkpoint_abs_errors.append(abs_error(checkpoint_action, teacher_action))

        comparisons.append({
            "sample_index": int(sample_index),
            "episode_index": int(sample["episode_index"].item()),
            "frame_index": int(sample["frame_index"].item()),
            "task": str(sample["task"]),
            "teacher_action_6d": teacher_action.tolist(),
            "base_action_6d": base_action.tolist(),
            "checkpoint_action_6d": checkpoint_action.tolist(),
            "base_l2_6d": base_l2_6d,
            "checkpoint_l2_6d": checkpoint_l2_6d,
            "improvement_l2_6d": base_l2_6d - checkpoint_l2_6d,
            "base_l2_real_4d": base_l2_real_4d,
            "checkpoint_l2_real_4d": checkpoint_l2_real_4d,
            "improvement_l2_real_4d": base_l2_real_4d - checkpoint_l2_real_4d,
            "base_l2_padding_2d": base_l2_padding_2d,
            "checkpoint_l2_padding_2d": checkpoint_l2_padding_2d,
            "improvement_l2_padding_2d": base_l2_padding_2d - checkpoint_l2_padding_2d,
            "base_postprocess_status": base_results[i]["postprocess_status"],
            "checkpoint_postprocess_status": checkpoint_results[i]["postprocess_status"],
        })

    mean_base_l2_6d = float(np.mean(base_l2_6d_values))
    mean_checkpoint_l2_6d = float(np.mean(checkpoint_l2_6d_values))
    mean_base_l2_real_4d = float(np.mean(base_l2_real_4d_values))
    mean_checkpoint_l2_real_4d = float(np.mean(checkpoint_l2_real_4d_values))

    result = {
        "ok": True,
        "purpose": "Compare SmolVLA base and trained checkpoint against PandaPickAndPlace TQC teacher actions",
        "note": (
            "This checks offline action proximity on dataset samples only. "
            "It does not prove rollout success in PandaPickAndPlace-v3."
        ),
        "device": device,
        "repo_id": args.repo_id,
        "root": args.root,
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "num_frames": int(dataset.num_frames),
        "num_episodes": int(dataset.num_episodes),
        "used_dataset_stats": dataset_stats is not None,
        "sample_indices": sample_indices,
        "num_samples": len(sample_indices),
        "mean_base_l2_6d": mean_base_l2_6d,
        "mean_checkpoint_l2_6d": mean_checkpoint_l2_6d,
        "mean_improvement_l2_6d_positive_is_better": mean_base_l2_6d - mean_checkpoint_l2_6d,
        "mean_base_l2_real_4d": mean_base_l2_real_4d,
        "mean_checkpoint_l2_real_4d": mean_checkpoint_l2_real_4d,
        "mean_improvement_l2_real_4d_positive_is_better": mean_base_l2_real_4d - mean_checkpoint_l2_real_4d,
        "mean_base_l2_padding_2d": float(np.mean(base_l2_padding_2d_values)),
        "mean_checkpoint_l2_padding_2d": float(np.mean(checkpoint_l2_padding_2d_values)),
        "base_l2_6d_summary": summarize_values(base_l2_6d_values),
        "checkpoint_l2_6d_summary": summarize_values(checkpoint_l2_6d_values),
        "base_l2_real_4d_summary": summarize_values(base_l2_real_4d_values),
        "checkpoint_l2_real_4d_summary": summarize_values(checkpoint_l2_real_4d_values),
        "base_abs_error_per_dim": summarize_per_dim(base_abs_errors),
        "checkpoint_abs_error_per_dim": summarize_per_dim(checkpoint_abs_errors),
        "comparisons": comparisons,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "offline_comparison_summary.json"
    csv_path = out_dir / "offline_comparison_rows.csv"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(csv_path, comparisons)

    print(json.dumps(result, indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()

