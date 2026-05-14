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


def build_features(use_videos: bool) -> dict[str, dict[str, Any]]:
    image_dtype = "video" if use_videos else "image"

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (24,),
            "names": [
                "obs_0",
                "obs_1",
                "obs_2",
                "obs_3",
                "obs_4",
                "obs_5",
                "obs_6",
                "obs_7",
                "obs_8",
                "obs_9",
                "obs_10",
                "obs_11",
                "obs_12",
                "obs_13",
                "obs_14",
                "obs_15",
                "obs_16",
                "obs_17",
                "achieved_goal_x",
                "achieved_goal_y",
                "achieved_goal_z",
                "desired_goal_x",
                "desired_goal_y",
                "desired_goal_z",
            ],
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

    if len(record["state"]) != 24:
        raise ValueError(f"Invalid state length: {len(record['state'])}")

    if len(record["action_6d"]) != 6:
        raise ValueError(f"Invalid action_6d length: {len(record['action_6d'])}")

    if not record.get("instruction"):
        raise ValueError("Missing instruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PandaPush visual teacher dataset to LeRobot format."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Raw PandaPush teacher dataset directory.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Output root for the local LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/panda_push_sac_her_teacher",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--robot-type", default="panda_gym_panda_push")
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

                for image_key in IMAGE_KEYS:
                    frame[image_key] = image

                dataset.add_frame(frame)
                total_frames += 1

            dataset.save_episode()
            print(f"saved_episode={episode_dir.name} frames={len(records)}")

    finally:
        dataset.finalize()

    loaded = LeRobotDataset(args.repo_id, root=root)

    result = {
        "input_dir": str(input_dir),
        "root": str(root),
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "fps": args.fps,
        "raw_accepted_episodes": metadata.get("accepted_episodes"),
        "raw_total_frames": metadata.get("total_frames"),
        "converted_episodes": loaded.num_episodes,
        "converted_frames": loaded.num_frames,
        "counted_frames": total_frames,
        "features": list(loaded.features.keys()),
    }

    print(json.dumps(result, indent=2))

    if result["converted_episodes"] != metadata.get("accepted_episodes"):
        raise RuntimeError("Episode count mismatch after conversion")

    if result["converted_frames"] != metadata.get("total_frames"):
        raise RuntimeError("Frame count mismatch after conversion")

    print("LEROBOT_CONVERSION_OK")


if __name__ == "__main__":
    main()
