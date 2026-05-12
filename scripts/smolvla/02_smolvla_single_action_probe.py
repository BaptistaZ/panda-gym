import json
from pathlib import Path

import gymnasium as gym
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


def map_smolvla_to_panda_action(raw_action, action_space, scale=0.05):
    action = raw_action.detach().cpu().numpy().reshape(-1)
    mapped = action[: action_space.shape[0]]
    mapped = np.tanh(mapped) * scale
    mapped = np.clip(mapped, action_space.low, action_space.high)
    return mapped.astype(np.float32)


def main():
    out_dir = Path("outputs/smolvla_probe/single_action")
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = "pick up the cube"
    env_id = "PandaPickAndPlace-v3"
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    frame = env.render()

    state_6d = np.concatenate(
        [obs["achieved_goal"], obs["desired_goal"]]
    ).astype(np.float32)

    model_id = "lerobot/smolvla_base"
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    policy.config.device = device

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        dataset_stats=None,
    )

    image_tensor = frame_to_tensor(frame)

    raw_batch = {
        "observation.state": torch.tensor(state_6d, dtype=torch.float32),
        "observation.images.camera1": image_tensor,
        "observation.images.camera2": image_tensor,
        "observation.images.camera3": image_tensor,
        "task": instruction,
    }

    processed = preprocessor(raw_batch)

    with torch.no_grad():
        raw_action = policy.select_action(processed)

    panda_action = map_smolvla_to_panda_action(
        raw_action,
        env.action_space,
        scale=0.05,
    )

    next_obs, reward, terminated, truncated, step_info = env.step(panda_action)
    next_frame = env.render()

    Image.fromarray(frame).save(out_dir / "before.png")
    Image.fromarray(next_frame).save(out_dir / "after.png")

    result = {
        "env_id": env_id,
        "seed": seed,
        "instruction": instruction,
        "device": device,
        "state_mapping": "concat(achieved_goal, desired_goal)",
        "state_6d": state_6d.tolist(),
        "raw_smolvla_action_6d": raw_action.detach().cpu().numpy().reshape(-1).tolist(),
        "mapped_panda_action_4d": panda_action.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": step_info,
        "next_achieved_goal": next_obs["achieved_goal"].tolist(),
        "next_desired_goal": next_obs["desired_goal"].tolist(),
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_dir}")

    env.close()


if __name__ == "__main__":
    main()
