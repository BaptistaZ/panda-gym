#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
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
    observation_names = [f"observation_{idx}" for idx in range(32)]
    achieved_goal_names = [f"achieved_goal_{idx}" for idx in range(6)]
    desired_goal_names = [f"desired_goal_{idx}" for idx in range(6)]
    return observation_names + achieved_goal_names + desired_goal_names


def build_features(use_videos: bool) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (44,),
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


def load_episode_records(data_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def validate_record(record: dict[str, Any], episode_dir: Path) -> None:
    image_path = episode_dir / record["image_path"]

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if len(record["state"]) != 44:
        raise ValueError(f"Invalid state length: {len(record['state'])}. Expected 44.")

    if len(record["action"]) != 4:
        raise ValueError(f"Invalid raw action length: {len(record['action'])}. Expected 4.")

    if len(record["action_6d"]) != 6:
        raise ValueError(f"Invalid action_6d length: {len(record['action_6d'])}. Expected 6.")

    if not record.get("instruction"):
        raise ValueError("Missing instruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw PandaStack-v1 TQC teacher dataset to LeRobot format."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Raw PandaStack-v1 teacher dataset directory.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Output root for the local LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/panda_stack_v1_tqc_teacher_20",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--robot-type", default="panda_gym_panda_stack_v1")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    root = args.root

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")

    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("env_id") != "PandaStack-v1":
        raise ValueError(f"Unexpected env_id: {metadata.get('env_id')}")

    if int(metadata.get("expected_state_shape", -1)) != 44:
        raise ValueError(f"Unexpected expected_state_shape: {metadata.get('expected_state_shape')}")

    if int(metadata.get("raw_action_shape", -1)) != 4:
        raise ValueError(f"Unexpected raw_action_shape: {metadata.get('raw_action_shape')}")

    if int(metadata.get("stored_action_shape", -1)) != 6:
        raise ValueError(f"Unexpected stored_action_shape: {metadata.get('stored_action_shape')}")

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

    episode_dirs = sorted((input_dir / "episodes").glob("episode_*"))
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found in {input_dir / 'episodes'}")

    total_frames = 0

    try:
        for episode_dir in episode_dirs:
            data_path = episode_dir / "data.jsonl"
            if not data_path.exists():
                raise FileNotFoundError(f"Missing data.jsonl: {data_path}")

            records = load_episode_records(data_path)
            if not records:
                raise RuntimeError(f"No frame records found in {data_path}")

            records = sorted(records, key=lambda item: int(item["step_index"]))

            for record in records:
                validate_record(record, episode_dir)

                image = load_image_256(episode_dir / record["image_path"])
                state = np.asarray(record["state"], dtype=np.float32)
                action = np.asarray(record["action_6d"], dtype=np.float32)
                task = str(record["instruction"])

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
            print(f"saved_episode={episode_dir.name} frames={len(records)}")

    finally:
        dataset.finalize()

    loaded = LeRobotDataset(args.repo_id, root=root)

    expected_episodes = int(metadata.get("episodes"))
    expected_frames = int(metadata.get("total_frames"))

    result = {
        "input_dir": str(input_dir),
        "root": str(root),
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "fps": args.fps,
        "raw_episodes": expected_episodes,
        "raw_total_frames": expected_frames,
        "raw_success_rate": metadata.get("success_rate"),
        "raw_mean_improvement": metadata.get("mean_improvement"),
        "converted_episodes": int(loaded.num_episodes),
        "converted_frames": int(loaded.num_frames),
        "counted_frames": total_frames,
        "features": list(loaded.features.keys()),
    }

    print(json.dumps(result, indent=2))

    if result["converted_episodes"] != expected_episodes:
        raise RuntimeError("Episode count mismatch after conversion")

    if result["converted_frames"] != expected_frames:
        raise RuntimeError("Frame count mismatch after conversion")

    print("LEROBOT_CONVERSION_OK")


if __name__ == "__main__":
    main()
