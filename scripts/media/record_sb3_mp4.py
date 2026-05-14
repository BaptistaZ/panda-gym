import argparse
import os
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import panda_gym  # noqa: F401
from stable_baselines3 import PPO


def _safe_makedirs_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def record_demo(
    env_id: str,
    model_path: str,
    out_path: str,
    episodes: int,
    steps_per_episode: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    deterministic: bool,
    action_noise_std: float,
    action_scale: float,
    ema: float,
    hold_last_frames: int,
) -> None:
    _safe_makedirs_for_file(out_path)

    env = gym.make(env_id, render_mode="rgb_array", renderer="Tiny")
    model = PPO.load(model_path, device="cpu")

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

    try:
        prev_action = None

        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            robot = env.unwrapped.robot

            # frame inicial do episódio
            frame = robot.sim.render(width=width, height=height)
            writer.append_data(frame)

            done = False
            n = 0

            while not done and n < steps_per_episode:
                action, _ = model.predict(obs, deterministic=deterministic)

                # ruído opcional
                if action_noise_std > 0.0:
                    action = action + np.random.normal(0.0, action_noise_std, size=np.shape(action))

                # EMA (suaviza “solavancos”)
                if ema > 0.0:
                    if prev_action is None:
                        prev_action = np.array(action, dtype=np.float32)
                    action = ema * prev_action + (1.0 - ema) * np.array(action, dtype=np.float32)
                    prev_action = np.array(action, dtype=np.float32)

                # abrandar movimento no próprio controlo
                action = np.array(action, dtype=np.float32) * float(action_scale)

                # clamp aos limites do env
                action = np.clip(action, env.action_space.low, env.action_space.high)

                obs, r, term, trunc, info = env.step(action)
                done = bool(term or trunc)
                n += 1

                frame = robot.sim.render(width=width, height=height)
                writer.append_data(frame)

            # segurar no fim (para o olho perceber o “sucesso”)
            if hold_last_frames > 0:
                last = robot.sim.render(width=width, height=height)
                for _ in range(hold_last_frames):
                    writer.append_data(last)

            print(f"[EP {ep+1:02d}/{episodes}] steps={n} done={done}")

        print(f"[OK] wrote {out_path}")

    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=480)

    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--action-noise", type=float, default=0.0)

    # NOVO: controlo “natural”
    ap.add_argument("--action-scale", type=float, default=0.2, help="0.1–0.3 costuma dar movimento natural")
    ap.add_argument("--ema", type=float, default=0.85, help="0.0 desliga; 0.8–0.95 suaviza bem")
    ap.add_argument("--hold", type=int, default=30, help="frames extra no fim do episódio")

    args = ap.parse_args()

    record_demo(
        env_id=args.env,
        model_path=args.model,
        out_path=args.out,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        deterministic=args.deterministic,
        action_noise_std=args.action_noise,
        action_scale=args.action_scale,
        ema=args.ema,
        hold_last_frames=args.hold,
    )
