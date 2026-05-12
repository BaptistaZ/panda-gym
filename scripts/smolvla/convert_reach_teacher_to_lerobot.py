import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


STATE_COLUMNS = [f"state_{idx}" for idx in range(6)]
ACTION_COLUMNS = [f"teacher_action_6d_{idx}" for idx in range(6)]

IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def build_features(use_videos: bool) -> dict:
    image_dtype = "video" if use_videos else "image"

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "achieved_x",
                "achieved_y",
                "achieved_z",
                "desired_x",
                "desired_y",
                "desired_z",
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


def validate_columns(df: pd.DataFrame) -> None:
    required = [
        "episode",
        "step",
        "instruction",
        "frame_path",
        *STATE_COLUMNS,
        *ACTION_COLUMNS,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in teacher CSV: {missing}")


def resolve_frame_path(csv_path: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Empty frame_path found. Regenerate the teacher dataset with --save-frames.")

    path = Path(raw_path)

    if path.exists():
        return path

    candidate = csv_path.parent / raw_path
    if candidate.exists():
        return candidate

    parts = path.parts
    if "frames" in parts:
        frames_index = parts.index("frames")
        candidate = csv_path.parent / Path(*parts[frames_index:])
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Frame not found: {raw_path}")


def load_image_256(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")

    if image.size != (256, 256):
        image = image.resize((256, 256), resample=Image.BILINEAR)

    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to teacher_steps.csv")
    parser.add_argument("--root", required=True, help="Output root for the local LeRobot dataset")
    parser.add_argument("--repo-id", default="local/panda_reach_ppo_teacher")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--robot-type", default="panda_gym_panda_reach")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    root = Path(args.root)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {root}. "
                "Use --overwrite only if you intentionally want to replace it."
            )
        shutil.rmtree(root)

    df = pd.read_csv(csv_path)
    validate_columns(df)

    if df["frame_path"].isna().any():
        raise ValueError("At least one frame_path is missing. Regenerate with --save-frames.")

    df = df.sort_values(["episode", "step"]).reset_index(drop=True)

    features = build_features(use_videos=args.use_videos)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type=args.robot_type,
        use_videos=args.use_videos,
    )

    try:
        for episode, episode_df in df.groupby("episode", sort=True):
            for _, row in episode_df.iterrows():
                frame_path = resolve_frame_path(csv_path, row["frame_path"])
                image = load_image_256(frame_path)

                state = row[STATE_COLUMNS].to_numpy(dtype=np.float32)
                action = row[ACTION_COLUMNS].to_numpy(dtype=np.float32)
                task = str(row["instruction"])

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

            dataset.save_episode()
            print(f"saved_episode={episode} frames={len(episode_df)}")

    finally:
        dataset.finalize()

    loaded = LeRobotDataset(args.repo_id, root=root)

    print("LeRobot dataset created successfully")
    print(f"root={root}")
    print(f"repo_id={args.repo_id}")
    print(f"num_episodes={loaded.num_episodes}")
    print(f"num_frames={loaded.num_frames}")
    print(f"features={list(loaded.features.keys())}")


if __name__ == "__main__":
    main()
