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
    elif mapping_mode == "invert_xyz":
        mapped[:3] *= -1.0

    mapped = np.tanh(mapped) * scale
    mapped = np.clip(mapped, action_space.low, action_space.high)
    return mapped.astype(np.float32), action.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smolvla", "zero", "random", "ppo"], required=True)
    parser.add_argument("--env", default="PandaReach-v3")
    parser.add_argument("--instruction", default="reach the target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--action-scale", type=float, default=0.20)
    parser.add_argument("--mapping-mode", choices=["direct", "invert_x", "invert_y", "invert_z", "invert_xyz"], default="direct")
    parser.add_argument("--ppo-model", default="models/PandaReach/ee/final.zip")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(args.env, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
    env.action_space.seed(args.seed)
    obs, info = env.reset(seed=args.seed)

    pre_step_initial_distance = float(
        np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
    )

    smolvla_policy = None
    preprocessor = None
    ppo_model = None

    if args.mode == "smolvla":
        smolvla_policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
        smolvla_policy.config.device = device
        preprocessor, _ = make_pre_post_processors(smolvla_policy.config, dataset_stats=None)

    if args.mode == "ppo":
        from stable_baselines3 import PPO
        ppo_model = PPO.load(args.ppo_model, device=device)

    csv_path = out_dir / "actions.csv"
    frame_paths = []
    total_reward = 0.0
    success_seen = False

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
            "distance_to_target",
            "raw_action_0",
            "raw_action_1",
            "raw_action_2",
            "raw_action_3",
            "raw_action_4",
            "raw_action_5",
            "panda_action_0",
            "panda_action_1",
            "panda_action_2",
        ])

        for step in range(args.steps):
            frame = env.render()
            frame_path = frames_dir / f"frame_{step:04d}.png"
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)

            if args.mode == "smolvla":
                state_6d = build_state_6d(obs)
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
                    raw_action_tensor = smolvla_policy.select_action(processed)

                panda_action, raw_action = map_smolvla_to_panda_action(
                    raw_action_tensor,
                    env.action_space,
                    args.action_scale,
                    args.mapping_mode,
                )

            elif args.mode == "ppo":
                panda_action, _ = ppo_model.predict(obs, deterministic=True)
                panda_action = np.asarray(panda_action, dtype=np.float32)
                raw_action = np.full(6, np.nan, dtype=np.float32)

            elif args.mode == "zero":
                panda_action = np.zeros(env.action_space.shape, dtype=np.float32)
                raw_action = np.zeros(6, dtype=np.float32)

            else:
                panda_action = env.action_space.sample().astype(np.float32)
                raw_action = np.full(6, np.nan, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(panda_action)

            achieved = obs["achieved_goal"]
            desired = obs["desired_goal"]
            distance = float(np.linalg.norm(achieved - desired))

            total_reward += float(reward)
            is_success = bool(info.get("is_success", False))
            success_seen = success_seen or is_success

            writer.writerow([
                step,
                args.mode,
                float(reward),
                is_success,
                *achieved.tolist(),
                *desired.tolist(),
                distance,
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

    import pandas as pd
    df = pd.read_csv(csv_path)
    distances = df["distance_to_target"].to_numpy()
    best_step = int(np.argmin(distances))

    summary = {
        "env_id": args.env,
        "mode": args.mode,
        "instruction": args.instruction,
        "seed": args.seed,
        "steps_requested": args.steps,
        "max_episode_steps": args.max_episode_steps,
        "steps_recorded": int(len(df)),
        "action_scale": args.action_scale,
        "mapping_mode": args.mapping_mode,
        "device": device,
        "pre_step_initial_distance": pre_step_initial_distance,
        "initial_distance": float(distances[0]),
        "best_distance": float(distances[best_step]),
        "best_step": best_step,
        "final_distance": float(distances[-1]),
        "improvement": float(distances[0] - distances[best_step]),
        "success_seen": success_seen,
        "total_reward": total_reward,
        "actions_csv": str(csv_path),
        "video_path": str(video_path),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
