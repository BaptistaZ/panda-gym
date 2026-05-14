import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import panda_gym
import torch
from PIL import Image

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def frame_to_tensor(frame, size=256):
    image = Image.fromarray(frame).resize((size, size))
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32)


def build_state_6d(obs):
    return np.concatenate([obs["achieved_goal"], obs["desired_goal"]]).astype(np.float32)


def map_smolvla_to_panda_action(raw_action, action_space, scale, mapping_mode):
    action = raw_action.detach().cpu().numpy().reshape(-1)
    mapped = action[: action_space.shape[0]].copy()

    if mapping_mode == "invert_x":
        mapped[0] *= -1.0
    elif mapping_mode == "invert_y":
        mapped[1] *= -1.0
    elif mapping_mode == "invert_z":
        mapped[2] *= -1.0
    elif mapping_mode == "invert_xy":
        mapped[0] *= -1.0
        mapped[1] *= -1.0
    elif mapping_mode == "invert_xz":
        mapped[0] *= -1.0
        mapped[2] *= -1.0
    elif mapping_mode == "invert_yz":
        mapped[1] *= -1.0
        mapped[2] *= -1.0
    elif mapping_mode == "invert_xyz":
        mapped[:3] *= -1.0

    mapped = np.tanh(mapped) * scale
    mapped = np.clip(mapped, action_space.low, action_space.high)
    return mapped.astype(np.float32), action.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PandaPickAndPlace-v3")
    parser.add_argument("--instruction", default="pick up the cube")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--action-scale", type=float, default=0.05)
    parser.add_argument("--mode", choices=["smolvla", "zero", "random"], default="smolvla")
    parser.add_argument(
        "--mapping-mode",
        choices=[
            "direct",
            "invert_x",
            "invert_y",
            "invert_z",
            "invert_xy",
            "invert_xz",
            "invert_yz",
            "invert_xyz",
        ],
        default="direct",
    )
    parser.add_argument("--out-dir", default="outputs/smolvla_probe/rollout_smolvla_seed0")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_kwargs = {"render_mode": "rgb_array"}
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(args.env, **env_kwargs)
    obs, info = env.reset(seed=args.seed)

    policy = None
    preprocessor = None

    if args.mode == "smolvla":
        model_id = "lerobot/smolvla_base"
        policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
        policy.config.device = device
        preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=None)
    else:
        model_id = None

    csv_path = out_dir / "actions.csv"
    frame_paths = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "mode",
            "reward",
            "is_success",
            "achieved_x",
            "achieved_y",
            "achieved_z",
            "desired_x",
            "desired_y",
            "desired_z",
            "ee_x",
            "ee_y",
            "ee_z",
            "raw_action_0",
            "raw_action_1",
            "raw_action_2",
            "raw_action_3",
            "raw_action_4",
            "raw_action_5",
            "panda_action_0",
            "panda_action_1",
            "panda_action_2",
            "panda_action_3",
        ])

        total_reward = 0.0
        success_seen = False

        for step in range(args.steps):
            frame = env.render()
            frame_path = frames_dir / f"frame_{step:04d}.png"
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)

            state_6d = build_state_6d(obs)

            if args.mode == "smolvla":
                image_tensor = frame_to_tensor(frame)

                raw_batch = {
                    "observation.state": torch.tensor(state_6d, dtype=torch.float32),
                    "observation.images.camera1": image_tensor,
                    "observation.images.camera2": image_tensor,
                    "observation.images.camera3": image_tensor,
                    "task": args.instruction,
                }

                processed = preprocessor(raw_batch)

                with torch.no_grad():
                    raw_action_tensor = policy.select_action(processed)

                panda_action, raw_action = map_smolvla_to_panda_action(
                    raw_action_tensor,
                    env.action_space,
                    args.action_scale,
                    args.mapping_mode,
                )

            elif args.mode == "zero":
                raw_action = np.zeros(6, dtype=np.float32)
                panda_action = np.zeros(env.action_space.shape, dtype=np.float32)

            else:
                raw_action = np.full(6, np.nan, dtype=np.float32)
                panda_action = env.action_space.sample().astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(panda_action)

            total_reward += float(reward)
            is_success = bool(info.get("is_success", False))
            success_seen = success_seen or is_success

            achieved = obs["achieved_goal"]
            desired = obs["desired_goal"]
            ee_position = np.asarray(env.unwrapped.robot.get_ee_position(), dtype=np.float32)

            writer.writerow([
                step,
                args.mode,
                float(reward),
                is_success,
                *achieved.tolist(),
                *desired.tolist(),
                *ee_position.tolist(),
                *raw_action.tolist(),
                *panda_action.tolist(),
            ])

            if terminated or truncated:
                break

    final_frame = env.render()
    final_frame_path = frames_dir / f"frame_{len(frame_paths):04d}.png"
    Image.fromarray(final_frame).save(final_frame_path)
    frame_paths.append(final_frame_path)

    video_path = out_dir / "rollout.mp4"
    with imageio.get_writer(video_path, fps=args.fps) as writer:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))

    summary = {
        "env_id": args.env,
        "mode": args.mode,
        "instruction": args.instruction,
        "seed": args.seed,
        "steps_requested": args.steps,
        "max_episode_steps": args.max_episode_steps,
        "steps_recorded": len(frame_paths),
        "action_scale": args.action_scale,
        "mapping_mode": args.mapping_mode,
        "device": device,
        "model_id": model_id,
        "total_reward": total_reward,
        "success_seen": success_seen,
        "actions_csv": str(csv_path),
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
