from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def checkpoint_label(checkpoint: str) -> str:
    path = Path(checkpoint)
    if path.name == "pretrained_model":
        return path.parent.name
    return path.name


def to_numpy_action(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    else:
        value = np.asarray(value, dtype=np.float32)

    value = np.asarray(value, dtype=np.float32)

    if value.ndim == 0:
        value = value.reshape(1)
    elif value.ndim == 1:
        pass
    elif value.ndim == 2:
        value = value[0]
    elif value.ndim == 3:
        value = value[0, 0]
    else:
        value = value.reshape(-1)

    return value.astype(np.float32).reshape(-1)


def get_task_text(sample: dict) -> str:
    task = sample.get("task", "push the cube to the target")

    if isinstance(task, torch.Tensor):
        if task.ndim == 0:
            return str(task.item())
        return str(task.detach().cpu().tolist())

    return str(task)


def get_processor(policy):
    candidates = [
        getattr(getattr(getattr(policy, "model", None), "vlm_with_expert", None), "processor", None),
        getattr(getattr(policy, "model", None), "processor", None),
        getattr(policy, "processor", None),
    ]

    for candidate in candidates:
        if candidate is not None:
            return candidate

    raise RuntimeError("Could not find a tokenizer/processor inside the SmolVLA policy.")


def add_language_tokens(batch: dict, policy, task: str, device: str) -> dict:
    processor = get_processor(policy)
    max_length = int(getattr(policy.config, "tokenizer_max_length", 48))

    if hasattr(processor, "tokenizer"):
        encoded = processor.tokenizer(
            [task],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        encoded = processor(
            text=[task],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    batch[OBS_LANGUAGE_TOKENS] = encoded["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"].to(device).bool()

    return batch


def build_batch(sample: dict, policy, device: str) -> dict:
    batch = {}

    for key, value in sample.items():
        if key.startswith("observation.") and isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)

    task = get_task_text(sample)
    batch["task"] = [task]

    batch = add_language_tokens(batch, policy, task, device)
    return batch


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sign(a) == np.sign(b)))


def compute_metrics(pred: np.ndarray, teacher: np.ndarray) -> dict:
    common_dim = min(len(pred), len(teacher))
    pred_common = pred[:common_dim]
    teacher_common = teacher[:common_dim]

    diff = pred_common - teacher_common

    metrics = {
        "pred_dim": int(len(pred)),
        "teacher_dim": int(len(teacher)),
        "common_dim": int(common_dim),
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "cosine_similarity": cosine_similarity(pred_common, teacher_common),
        "teacher_norm": float(np.linalg.norm(teacher_common)),
        "pred_norm": float(np.linalg.norm(pred_common)),
        "norm_ratio": float(np.linalg.norm(pred_common) / (np.linalg.norm(teacher_common) + 1e-12)),
        "sign_agreement": sign_agreement(pred_common, teacher_common),
    }

    if common_dim >= 3:
        pred_3 = pred_common[:3]
        teacher_3 = teacher_common[:3]
        diff_3 = pred_3 - teacher_3
        metrics.update(
            {
                "mse_first3": float(np.mean(diff_3 ** 2)),
                "mae_first3": float(np.mean(np.abs(diff_3))),
                "rmse_first3": float(np.sqrt(np.mean(diff_3 ** 2))),
                "cosine_similarity_first3": cosine_similarity(pred_3, teacher_3),
                "teacher_norm_first3": float(np.linalg.norm(teacher_3)),
                "pred_norm_first3": float(np.linalg.norm(pred_3)),
                "norm_ratio_first3": float(np.linalg.norm(pred_3) / (np.linalg.norm(teacher_3) + 1e-12)),
                "sign_agreement_first3": sign_agreement(pred_3, teacher_3),
            }
        )

    return metrics


def aggregate(rows: list[dict]) -> dict:
    metric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"checkpoint", "frame_index", "dataset_index"}
        and isinstance(rows[0][key], (int, float))
    ]

    summary = {"num_samples": len(rows)}
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"std_{key}"] = float(np.std(values))

    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )

    num_frames = len(dataset)
    if args.max_samples <= 0 or args.max_samples >= num_frames:
        indices = list(range(num_frames))
    else:
        indices = sorted(set(np.linspace(0, num_frames - 1, args.max_samples, dtype=int).tolist()))

    print(f"dataset_frames={num_frames}")
    print(f"selected_samples={len(indices)}")
    print(f"device={args.device}")

    all_summaries = {}

    for checkpoint in args.checkpoints:
        label = checkpoint_label(checkpoint)
        print("=" * 80)
        print(f"checkpoint={label}")
        print(f"path={checkpoint}")

        policy = SmolVLAPolicy.from_pretrained(checkpoint).to(args.device).eval()

        rows = []

        for local_i, dataset_i in enumerate(indices):
            sample = dataset[dataset_i]
            teacher_action = to_numpy_action(sample["action"])
            batch = build_batch(sample, policy, args.device)

            if hasattr(policy, "reset"):
                policy.reset()

            with torch.inference_mode():
                pred_action = policy.select_action(batch)

            pred_action = to_numpy_action(pred_action)

            metrics = compute_metrics(pred_action, teacher_action)
            row = {
                "checkpoint": label,
                "frame_index": int(local_i),
                "dataset_index": int(dataset_i),
                **metrics,
            }
            rows.append(row)

        summary = aggregate(rows)
        all_summaries[label] = summary

        checkpoint_output = output_dir / label
        checkpoint_output.mkdir(parents=True, exist_ok=True)

        with (checkpoint_output / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with (checkpoint_output / "predictions_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(json.dumps(summary, indent=2))

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_dir / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 80)
    print("ACTION_IMITATION_DIAGNOSTIC_OK")
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
