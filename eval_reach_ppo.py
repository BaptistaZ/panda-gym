import argparse
import numpy as np
import gymnasium as gym
import panda_gym  # noqa: F401
from stable_baselines3 import PPO


def eval_model(env_id: str, model_path: str, episodes: int, seed: int, deterministic: bool = True):
    env = gym.make(env_id)
    model = PPO.load(model_path, device="cpu")

    returns = []
    lengths = []
    successes = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_ret += float(r)
            ep_len += 1
            # goal-env style
            if isinstance(info, dict) and "is_success" in info:
                ep_success = float(info["is_success"])

        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(ep_success)

    env.close()

    return {
        "episodes": episodes,
        "avg_return": float(np.mean(returns)),
        "avg_len": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true", help="se quiseres avaliar com ação não-determinística")
    args = ap.parse_args()

    res = eval_model(args.env, args.model, args.episodes, args.seed, deterministic=(not args.stochastic))
    print(f"env={args.env}")
    print(f"model={args.model}")
    print(f"episodes={res['episodes']} avg_return={res['avg_return']:.3f} success_rate={res['success_rate']:.3f} avg_len={res['avg_len']:.1f}")
