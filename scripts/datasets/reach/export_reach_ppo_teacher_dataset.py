import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import panda_gym
from PIL import Image
from stable_baselines3 import PPO


def build_state_6d(obs):
    return np.concatenate([obs["achieved_goal"], obs["desired_goal"]]).astype(np.float32)


def pad_action_to_6d(action):
    padded = np.zeros(6, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    padded[: min(len(action), 6)] = action[:6]
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PandaReach-v3")
    parser.add_argument("--instruction", default="reach the target")
    parser.add_argument("--ppo-model", default="models/PandaReach/ee/final.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/teacher_datasets/panda_reach_ppo")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_frames or args.save_video:
        frames_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.ppo_model)
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")

    env = gym.make(args.env, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
    model = PPO.load(str(model_path), device="auto")

    csv_path = out_dir / "teacher_steps.csv"
    episodes_summary = []
    all_video_frames = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "episode",
            "step",
            "instruction",
            "reward",
            "is_success",
            "distance_to_target",
            "achieved_x",
            "achieved_y",
            "achieved_z",
            "desired_x",
            "desired_y",
            "desired_z",
            "state_0",
            "state_1",
            "state_2",
            "state_3",
            "state_4",
            "state_5",
            "ppo_action_0",
            "ppo_action_1",
            "ppo_action_2",
            "teacher_action_6d_0",
            "teacher_action_6d_1",
            "teacher_action_6d_2",
            "teacher_action_6d_3",
            "teacher_action_6d_4",
            "teacher_action_6d_5",
            "next_distance_to_target",
            "frame_path",
        ]
        writer.writerow(header)

        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            obs, info = env.reset(seed=episode_seed)

            initial_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            best_distance = initial_distance
            success_seen = False
            total_reward = 0.0
            steps = 0

            for step in range(args.max_episode_steps):
                # This row stores observation_t/frame_t paired with teacher action_t.
                frame = env.render()
                frame_path = ""

                if args.save_frames or args.save_video:
                    frame_file = frames_dir / f"episode_{episode:03d}_step_{step:03d}.png"
                    Image.fromarray(frame).save(frame_file)
                    frame_path = str(frame_file)

                if args.save_video:
                    all_video_frames.append(frame)

                achieved = obs["achieved_goal"].astype(np.float32)
                desired = obs["desired_goal"].astype(np.float32)
                state_6d = build_state_6d(obs)
                distance = float(np.linalg.norm(achieved - desired))

                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                action_6d = pad_action_to_6d(action)

                next_obs, reward, terminated, truncated, info = env.step(action)

                next_distance = float(
                    np.linalg.norm(next_obs["achieved_goal"] - next_obs["desired_goal"])
                )
                is_success = bool(info.get("is_success", False))

                best_distance = min(best_distance, next_distance)
                success_seen = success_seen or is_success
                total_reward += float(reward)
                steps += 1

                writer.writerow([
                    episode,
                    step,
                    args.instruction,
                    float(reward),
                    is_success,
                    distance,
                    *achieved.tolist(),
                    *desired.tolist(),
                    *state_6d.tolist(),
                    *action[:3].tolist(),
                    *action_6d.tolist(),
                    next_distance,
                    frame_path,
                ])

                obs = next_obs

                if terminated or truncated:
                    break

            episodes_summary.append({
                "episode": episode,
                "seed": episode_seed,
                "initial_distance": initial_distance,
                "best_distance": best_distance,
                "improvement": initial_distance - best_distance,
                "success_seen": success_seen,
                "total_reward": total_reward,
                "steps": steps,
            })

    video_path = None
    if args.save_video:
        video_path = out_dir / "teacher_rollouts.mp4"
        with imageio.get_writer(video_path, fps=args.fps) as writer:
            for frame in all_video_frames:
                writer.append_data(frame)

    success_rate = sum(ep["success_seen"] for ep in episodes_summary) / len(episodes_summary)

    summary = {
        "env_id": args.env,
        "instruction": args.instruction,
        "ppo_model": str(model_path),
        "episodes": args.episodes,
        "max_episode_steps": args.max_episode_steps,
        "seed_start": args.seed,
        "csv_path": str(csv_path),
        "frames_dir": str(frames_dir) if (args.save_frames or args.save_video) else None,
        "video_path": str(video_path) if video_path else None,
        "success_rate": success_rate,
        "mean_best_distance": float(np.mean([ep["best_distance"] for ep in episodes_summary])),
        "mean_improvement": float(np.mean([ep["improvement"] for ep in episodes_summary])),
        "episodes_summary": episodes_summary,
        "alignment_note": (
            "Rows store observation_t/frame_t paired with teacher action_t. "
            "next_distance_to_target is measured after env.step(action_t)."
        ),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
