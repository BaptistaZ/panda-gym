#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
import socketserver
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


IMAGE_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def decode_image_b64(image_b64: str) -> torch.Tensor:
    image_bytes = base64.b64decode(image_b64.encode("utf-8"))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).contiguous()


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


class PandaStackActionModel:
    def __init__(
        self,
        checkpoint: str | Path,
        dataset_root: str | Path,
        repo_id: str,
        device: str,
    ) -> None:
        self.device = device
        self.dataset = LeRobotDataset(repo_id, root=Path(dataset_root))
        self.dataset_stats = self.dataset.meta.stats

        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
        self.policy.config.device = device

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            dataset_stats=self.dataset_stats,
        )

    def act(self, state: list[float], image_b64: str, task: str) -> dict[str, Any]:
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)

        if state_np.shape[0] != 32:
            raise ValueError(f"Expected state length 32, got {state_np.shape[0]}")

        image = decode_image_b64(image_b64)

        raw_batch: dict[str, Any] = {
            "observation.state": to_tensor(state_np),
            "task": str(task),
        }

        for key in IMAGE_KEYS:
            raw_batch[key] = image.detach().clone()

        processed = self.preprocessor(raw_batch)

        for key, value in list(processed.items()):
            if isinstance(value, torch.Tensor):
                processed[key] = value.to(self.device)

        if hasattr(self.policy, "reset"):
            self.policy.reset()

        with torch.no_grad():
            raw_action = self.policy.select_action(processed)
            final_action = maybe_postprocess_action(self.postprocessor, raw_action)

        action_6d = final_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        action_4d = action_6d[:4].astype(np.float32)

        return {
            "ok": True,
            "action_4d": action_4d.tolist(),
            "action_6d": action_6d.tolist(),
        }


class ActionRequestHandler(socketserver.StreamRequestHandler):
    model: PandaStackActionModel

    def handle(self) -> None:
        line = self.rfile.readline().decode("utf-8").strip()

        try:
            request = json.loads(line)

            response = self.model.act(
                state=request["state"],
                image_b64=request["image_png_b64"],
                task=request.get("task", "stack the blocks"),
            )

        except Exception as exc:
            response = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def encode_image_file(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_self_test(model: PandaStackActionModel, raw_dataset_root: Path) -> None:
    episode_dir = raw_dataset_root / "episodes" / "episode_000000"
    data_path = episode_dir / "data.jsonl"

    first_record = json.loads(data_path.read_text(encoding="utf-8").splitlines()[0])

    state = first_record["observation"]
    image_path = episode_dir / first_record["image_path"]
    image_b64 = encode_image_file(image_path)

    response = model.act(
        state=state,
        image_b64=image_b64,
        task=first_record.get("instruction", "stack the blocks"),
    )

    print("SERVER_SELF_TEST_OK")
    print("action_4d_len:", len(response["action_4d"]))
    print("action_6d_len:", len(response["action_6d"]))
    print("action_4d:", response["action_4d"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve SmolVLA PandaStack-v1 actions over localhost."
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/train/smolvla_panda_stack_v1_state32_1000/checkpoints/000750/pretrained_model",
    )
    parser.add_argument(
        "--dataset-root",
        default="outputs/lerobot_datasets/panda_stack_v1_tqc_teacher_20_state32",
    )
    parser.add_argument(
        "--repo-id",
        default="local/panda_stack_v1_tqc_teacher_20_state32",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--raw-dataset-root",
        default="outputs/teacher_datasets/pandastack_v1_tqc_raw_20",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading SmolVLA PandaStack action model...")
    print("checkpoint:", args.checkpoint)
    print("dataset_root:", args.dataset_root)
    print("repo_id:", args.repo_id)
    print("device:", args.device)

    model = PandaStackActionModel(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        device=args.device,
    )

    if args.self_test:
        run_self_test(model, Path(args.raw_dataset_root))
        return

    ActionRequestHandler.model = model

    with ReusableTCPServer((args.host, args.port), ActionRequestHandler) as server:
        print(f"ACTION_SERVER_READY host={args.host} port={args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
