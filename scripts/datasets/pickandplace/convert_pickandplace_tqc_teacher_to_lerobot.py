#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def build_state_names() -> list[str]:
    observation_names = [f"observation_{idx}" for idx in range(19)]
    achieved_goal_names = [
        "achieved_goal_x",
        "achieved_goal_y",
        "achieved_goal_z",
    ]
    desired_goal_names = [
        "desired_goal_x",
        "desired_goal_y",
        "desired_goal_z",
    ]

    return observation_names + achieved_goal_names + desired_goal_names


def build_features(use_videos: bool) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (25,),
            "names": build_state_names(),
        },
        "observation.images.camera1": {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera3": {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "teacher_action_6d_0",
                "teacher_action_6d_1",
                "teacher_action_6d_2",
                "teacher_action_6d_3",
                "teacher_action_6d_4",
                "teacher_action_6d_5",
            ],
        },
    }


def load_image_256(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")

    if image.size != (256, 256):
        image = image.resize((256, 256), resample=Image.BILINEAR)

    return np.asarray(image, dtype=np.uint8)


def parse_json_vector(value: str, expected_len: int, field_name: str) -> np.ndarray:
    vector = np.asarray(json.loads(value), dtype=np.float32)

    if vector.shape != (expected_len,):
        raise ValueError(
            f"Invalid {field_name} shape: {vector.shape}. "
            f"Expected ({expected_len},)."
        )

    return vector


def resolve_frame_path(input_dir: Path, raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("Empty frame_path found")

    frame_path = Path(raw_path)

    if frame_path.is_absolute() and frame_path.exists():
        return frame_path

    candidate = input_dir / frame_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Frame not found: {candidate}")


def load_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    required_columns = {
        "episode",
        "step",
        "task",
        "frame_path",
        "state_25",
        "smolvla_action_6d",
    }

    missing = required_columns - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    if not records:
        raise RuntimeError(f"No records found in {csv_path}")

    return records


def validate_metadata(input_dir: Path) -> dict[str, Any]:
    metadata_path = input_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("env_id") != "PandaPickAndPlace-v3":
        raise ValueError(f"Unexpected env_id: {metadata.get('env_id')}")

    if metadata.get("state_shape") != [25]:
        raise ValueError(f"Unexpected state_shape: {metadata.get('state_shape')}")

    if metadata.get("smolvla_action_shape") != [6]:
        raise ValueError(
            f"Unexpected smolvla_action_shape: {metadata.get('smolvla_action_shape')}"
        )

    return metadata


def group_records_by_episode(records: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)

    for record in records:
        episode = int(record["episode"])
        grouped[episode].append(record)

    for episode, episode_records in grouped.items():
        episode_records.sort(key=lambda item: int(item["step"]))

    return dict(sorted(grouped.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw PandaPickAndPlace TQC teacher dataset to LeRobot format."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Raw PandaPickAndPlace teacher dataset directory.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Output root for the local LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/panda_pickandplace_tqc_teacher_10",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--robot-type", default="panda_gym_panda_pickandplace")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    root = args.root

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")

    metadata = validate_metadata(input_dir)

    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        raw_summary = {}

    csv_path = input_dir / "teacher_steps.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"teacher_steps.csv not found: {csv_path}")

    records = load_csv_records(csv_path)
    grouped_records = group_records_by_episode(records)

    if root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {root}. Use --overwrite to replace it."
            )
        shutil.rmtree(root)

    features = build_features(use_videos=args.use_videos)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
    )

    total_frames = 0

    try:
        for episode, episode_records in grouped_records.items():
            for record in episode_records:
                frame_path = resolve_frame_path(input_dir, record["frame_path"])
                image = load_image_256(frame_path)

                state = parse_json_vector(
                    record["state_25"],
                    expected_len=25,
                    field_name="state_25",
                )
                action = parse_json_vector(
                    record["smolvla_action_6d"],
                    expected_len=6,
                    field_name="smolvla_action_6d",
                )
                task = str(record["task"])

                frame = {
                    "observation.state": state,
                    "action": action,
                    "task": task,
                }

                # PandaGym provides one RGB render. It is replicated across the
                # three expected SmolVLA camera inputs for format compatibility.
                for image_key in IMAGE_KEYS:
                    frame[image_key] = image

                dataset.add_frame(frame)
                total_frames += 1

            dataset.save_episode()
            print(f"saved_episode={episode} frames={len(episode_records)}")

    finally:
        dataset.finalize()

    loaded = LeRobotDataset(args.repo_id, root=root)

    expected_episodes = int(metadata["episodes"])
    expected_frames = int(metadata["total_steps"])

    result = {
        "input_dir": str(input_dir),
        "root": str(root),
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "fps": args.fps,
        "raw_episodes": expected_episodes,
        "raw_total_frames": expected_frames,
        "converted_episodes": int(loaded.num_episodes),
        "converted_frames": int(loaded.num_frames),
        "counted_frames": total_frames,
        "features": list(loaded.features.keys()),
        "state_shape": [25],
        "action_shape": [6],
        "raw_success_rate": raw_summary.get("success_rate"),
    }

    print(json.dumps(result, indent=2))

    if result["converted_episodes"] != expected_episodes:
        raise RuntimeError("Episode count mismatch after conversion")

    if result["converted_frames"] != expected_frames:
        raise RuntimeError("Frame count mismatch after conversion")

    print("LEROBOT_CONVERSION_OK")


if __name__ == "__main__":
    main()

