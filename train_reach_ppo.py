import argparse
import os
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import panda_gym  # noqa: F401  (regista envs)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed


@dataclass(frozen=True)
class TrainConfig:
    env_id: str
    seed: int
    total_timesteps: int
    n_envs: int
    out_dir: str
    log_dir: str
    mid_name: str = "mid.zip"
    final_name: str = "final.zip"
    gamma: float = 0.98
    n_steps: int = 1024
    batch_size: int = 256
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    clip_range: float = 0.2


def make_env(env_id: str, rank: int, seed: int) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id)  # no render_mode: headless safe
        env = Monitor(env)
        # seed no reset (gymnasium)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


def main(cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    set_random_seed(cfg.seed)

    env = DummyVecEnv([make_env(cfg.env_id, i, cfg.seed) for i in range(cfg.n_envs)])
    env = VecMonitor(env)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        gamma=cfg.gamma,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        clip_range=cfg.clip_range,
        tensorboard_log=cfg.log_dir,
        verbose=1,
        seed=cfg.seed,
        device="cpu",
    )

    ten_pct = cfg.total_timesteps // 10
    half = cfg.total_timesteps // 2

    # 10%
    model.learn(total_timesteps=ten_pct, progress_bar=True, tb_log_name="ppo")
    ckpt10_path = os.path.join(cfg.out_dir, "ckpt10.zip")
    model.save(ckpt10_path)

    # 50%
    model.learn(total_timesteps=half - ten_pct, reset_num_timesteps=False, progress_bar=True, tb_log_name="ppo")
    mid_path = os.path.join(cfg.out_dir, cfg.mid_name)
    model.save(mid_path)

    # 100%
    model.learn(total_timesteps=cfg.total_timesteps - half, reset_num_timesteps=False, progress_bar=True, tb_log_name="ppo")
    final_path = os.path.join(cfg.out_dir, cfg.final_name)
    model.save(final_path)

    print(f"[OK] saved: {ckpt10_path}")
    print(f"[OK] saved: {mid_path}")
    print(f"[OK] saved: {final_path}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="ex: PandaReach-v3 ou PandaReachJoints-v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--out", required=True, help="pasta destino (ex: models/PandaReach/ee)")
    ap.add_argument("--log", required=True, help="pasta logs (ex: runs/PandaReach/ee)")
    args = ap.parse_args()

    main(
        TrainConfig(
            env_id=args.env,
            seed=args.seed,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            out_dir=args.out,
            log_dir=args.log,
        )
    )
