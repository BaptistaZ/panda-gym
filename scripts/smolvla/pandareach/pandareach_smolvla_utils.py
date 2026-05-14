from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


TASK_ALIASES = {
    "reach the target": "reach the target",
    "reach target": "reach the target",
    "go to target": "reach the target",
    "alcança o alvo": "reach the target",
    "alcanca o alvo": "reach the target",
    "alcançar o alvo": "reach the target",
    "alcancar o alvo": "reach the target",
    "chega ao alvo": "reach the target",
    "chegar ao alvo": "reach the target",
    "vai para o alvo": "reach the target",
}


def normalize_instruction(instruction: str) -> str:
    cleaned = " ".join(instruction.strip().lower().split())
    if not cleaned:
        return "reach the target"

    if cleaned not in TASK_ALIASES:
        accepted = ", ".join(sorted(TASK_ALIASES))
        raise ValueError(f"Unsupported instruction: {instruction!r}. Accepted instructions: {accepted}")

    return TASK_ALIASES[cleaned]


def to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def resize_rgb_to_chw_float(image: np.ndarray, size: int = 256) -> torch.Tensor:
    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")

    arr = np.ascontiguousarray(arr)
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).contiguous()


def build_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32).reshape(-1)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32).reshape(-1)

    if achieved.shape[0] != 3 or desired.shape[0] != 3:
        raise ValueError(f"Expected achieved_goal and desired_goal with shape 3, got {achieved.shape}, {desired.shape}")

    return np.concatenate([achieved, desired], axis=0).astype(np.float32)


def distance(obs: dict[str, np.ndarray]) -> float:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))


def maybe_postprocess_action(postprocessor: Any, action: torch.Tensor) -> torch.Tensor:
    if postprocessor is None:
        return action

    for candidate in [{"action": action}, action]:
        try:
            out = postprocessor(candidate)

            if isinstance(out, dict) and "action" in out:
                return out["action"]

            if isinstance(out, torch.Tensor):
                return out

        except Exception:
            pass

    return action


class SmolVLAPandaReachPolicy:
    def __init__(self, checkpoint: str | Path, dataset_stats: dict, device: str) -> None:
        self.device = device
        self.checkpoint = str(checkpoint)
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint).to(device).eval()
        self.policy.config.device = device

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            dataset_stats=dataset_stats,
        )

    def act(self, obs: dict[str, np.ndarray], rgb_image: np.ndarray, task: str = "reach the target") -> np.ndarray:
        state = build_state(obs)
        image = resize_rgb_to_chw_float(rgb_image)

        raw_batch: dict[str, Any] = {
            "observation.state": to_tensor(state),
            "task": task,
        }

        for key in IMAGE_KEYS:
            raw_batch[key] = image.detach().clone()

        processed = self.preprocessor(raw_batch)

        for key, value in list(processed.items()):
            if isinstance(value, torch.Tensor):
                processed[key] = value.to(self.device)

        with torch.no_grad():
            raw_action = self.policy.select_action(processed)
            final_action = maybe_postprocess_action(self.postprocessor, raw_action)

        return final_action.detach().cpu().numpy().reshape(-1)[:3].astype(np.float32)

    def close(self) -> None:
        del self.policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
