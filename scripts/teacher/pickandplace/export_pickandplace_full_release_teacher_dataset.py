from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
from PIL import Image

from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


TASK_TEXT = (
    "pick up the cube, place it on the target, "
    "release it, and move the gripper away"
)

OBJECT_Z = 0.02
HOVER_Z = 0.12

PRE_RELEASE_XY_TOLERANCE = 0.015
FINAL_XY_TOLERANCE = 0.04
FINAL_Z_TOLERANCE = 0.012

TEACHER_MAX_STEPS = 35
DESCENT_MAX_STEPS = 20
OPEN_STEPS = 4
RETREAT_STEPS = 6
SETTLE_MAX_STEPS = 5
REQUIRED_STABLE_STEPS = 5


def make_normalization_env():
    """Create the environment required for loading VecNormalize and TQC."""
    return gym.make(
        "PandaPickAndPlace-v3",
        render_mode="rgb_array",
        renderer="Tiny",
    )


def build_state_25(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Build the 25-dimensional state used by the existing dataset."""
    state = np.concatenate(
        [
            np.asarray(
                observation["observation"],
                dtype=np.float32,
            ),
            np.asarray(
                observation["achieved_goal"],
                dtype=np.float32,
            ),
            np.asarray(
                observation["desired_goal"],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)

    if state.shape != (25,):
        raise ValueError(
            f"Expected state shape (25,), got {state.shape}"
        )

    return state


def action_4d_to_smolvla_6d(
    action_4d: np.ndarray,
) -> np.ndarray:
    """Pad the Panda action to the six dimensions expected by SmolVLA."""
    action_4d = np.asarray(
        action_4d,
        dtype=np.float32,
    ).reshape(4)

    action_6d = np.zeros(
        6,
        dtype=np.float32,
    )

    action_6d[:4] = action_4d

    return action_6d


def save_rgb_frame(
    frame: np.ndarray,
    path: Path,
) -> None:
    """Save a rendered RGB frame as PNG."""
    if frame.dtype != np.uint8:
        frame = np.clip(
            frame,
            0,
            255,
        ).astype(np.uint8)

    Image.fromarray(frame).save(path)


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write formatted JSON."""
    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def raw_step(
    env: PandaPickAndPlaceEnv,
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Execute one simulation step without the original early termination."""
    env.robot.set_action(
        np.asarray(
            action,
            dtype=np.float32,
        )
    )

    env.sim.step()

    return env._get_obs()


def predict_teacher(
    model: TQC,
    normalizer: VecNormalize,
    observation: dict[str, np.ndarray],
    proxy_goal: np.ndarray,
) -> np.ndarray:
    """Predict the RL teacher action for an elevated proxy goal."""
    batch = {
        key: np.expand_dims(
            np.asarray(value),
            axis=0,
        )
        for key, value in observation.items()
    }

    batch["desired_goal"] = np.expand_dims(
        np.asarray(
            proxy_goal,
            dtype=np.float32,
        ),
        axis=0,
    )

    normalized_observation = normalizer.normalize_obs(
        batch
    )

    action, _ = model.predict(
        normalized_observation,
        deterministic=True,
    )

    return np.asarray(
        action,
        dtype=np.float32,
    ).reshape(4)


def cartesian_action(
    displacement: np.ndarray,
    gripper_command: float,
) -> np.ndarray:
    """Convert a Cartesian displacement into a normalized Panda action."""
    xyz_action = np.clip(
        np.asarray(
            displacement,
            dtype=np.float32,
        )
        / 0.05,
        -1.0,
        1.0,
    )

    return np.asarray(
        [
            xyz_action[0],
            xyz_action[1],
            xyz_action[2],
            gripper_command,
        ],
        dtype=np.float32,
    )


def contact_flags(
    env: PandaPickAndPlaceEnv,
) -> dict[str, bool]:
    """Read finger-object and object-table contact flags."""
    client = env.sim.physics_client
    bodies = env.sim._bodies_idx

    robot_object_contacts = client.getContactPoints(
        bodyA=bodies["panda"],
        bodyB=bodies["object"],
    )

    object_table_contacts = client.getContactPoints(
        bodyA=bodies["object"],
        bodyB=bodies["table"],
    )

    return {
        "left": any(
            int(contact[3]) == 9
            for contact in robot_object_contacts
        ),
        "right": any(
            int(contact[3]) == 10
            for contact in robot_object_contacts
        ),
        "table": bool(object_table_contacts),
    }


def full_success_metrics(
    env: PandaPickAndPlaceEnv,
    observation: dict[str, np.ndarray],
    final_goal: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate placement, release and physical stability."""
    cube_position = np.asarray(
        observation["achieved_goal"],
        dtype=np.float64,
    )

    cube_velocity = np.asarray(
        env.sim.get_base_velocity("object"),
        dtype=np.float64,
    )

    cube_angular_velocity = np.asarray(
        env.sim.get_base_angular_velocity("object"),
        dtype=np.float64,
    )

    fingers_width = float(
        observation["observation"][6]
    )

    contacts = contact_flags(env)

    xy_distance = float(
        np.linalg.norm(
            cube_position[:2]
            - final_goal[:2]
        )
    )

    z_error = abs(
        float(cube_position[2])
        - OBJECT_Z
    )

    linear_speed = float(
        np.linalg.norm(cube_velocity)
    )

    angular_speed = float(
        np.linalg.norm(
            cube_angular_velocity
        )
    )

    checks = {
        "xy_close": (
            xy_distance
            <= FINAL_XY_TOLERANCE
        ),
        "on_table": (
            z_error
            <= FINAL_Z_TOLERANCE
        ),
        "gripper_open": (
            fingers_width
            >= 0.06
        ),
        "released": (
            not contacts["left"]
            and not contacts["right"]
        ),
        "table_contact": contacts["table"],
        "stable_linear": (
            linear_speed
            <= 0.025
        ),
        "stable_angular": (
            angular_speed
            <= 0.5
        ),
    }

    metrics = {
        "cube_position": (
            cube_position.tolist()
        ),
        "xy_distance": xy_distance,
        "z_error": z_error,
        "fingers_width": fingers_width,
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
        "contacts": contacts,
        "checks": checks,
    }

    return all(checks.values()), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete PandaPickAndPlace demonstrations "
            "with grasp, transport, release, retreat and stability."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum number of consecutive seeds to try. "
            "Defaults to max(2 * episodes, episodes + 20)."
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "models/PandaPickAndPlace/"
            "chencliu_tqc_PandaPickAndPlace_v3"
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/teacher_datasets/"
            "panda_pickandplace_full_release_teacher_20"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero"
        )

    max_attempts = (
        args.max_attempts
        if args.max_attempts is not None
        else max(
            args.episodes * 2,
            args.episodes + 20,
        )
    )

    if max_attempts < args.episodes:
        raise ValueError(
            "--max-attempts must be greater than or equal to --episodes"
        )

    model_path = (
        args.model_dir
        / "tqc-PandaPickAndPlace-v3.zip"
    )

    vec_path = (
        args.model_dir
        / "vec_normalize.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {model_path}"
        )

    if not vec_path.exists():
        raise FileNotFoundError(
            f"Missing VecNormalize file: {vec_path}"
        )

    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: "
                f"{args.out_dir}. Use --overwrite."
            )

        shutil.rmtree(args.out_dir)

    frames_dir = args.out_dir / "frames"
    episodes_dir = args.out_dir / "episodes"
    staging_dir = args.out_dir / ".staging"

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    episodes_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalization_env = DummyVecEnv(
        [make_normalization_env]
    )

    normalizer = VecNormalize.load(
        str(vec_path),
        normalization_env,
    )

    normalizer.training = False
    normalizer.norm_reward = False

    model = TQC.load(
        str(model_path),
        env=normalization_env,
        device="cpu",
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
        },
    )

    env = PandaPickAndPlaceEnv(
        render_mode="rgb_array",
        renderer="Tiny",
    )

    csv_path = (
        args.out_dir
        / "teacher_steps.csv"
    )

    csv_fields = [
        "episode",
        "seed",
        "step",
        "phase",
        "task",
        "frame_path",
        "state_25",
        "env_action_4d",
        "smolvla_action_6d",
        "reward",
        "done",
        "is_success",
        "distance_to_goal",
    ]

    all_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []
    total_steps = 0
    attempted_seed_count = 0
    phase_frame_counts = {
        "teacher": 0,
        "descent": 0,
        "open": 0,
        "retreat": 0,
        "settle": 0,
    }

    try:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=csv_fields,
            )

            writer.writeheader()

            episode = 0

            while (
                episode < args.episodes
                and attempted_seed_count < max_attempts
            ):
                seed = (
                    args.seed_start
                    + attempted_seed_count
                )

                attempted_seed_count += 1

                observation, _ = env.reset(
                    seed=seed
                )

                final_goal = np.asarray(
                    observation["desired_goal"],
                    dtype=np.float64,
                )

                final_goal[2] = OBJECT_Z

                env.task.goal = (
                    final_goal.copy()
                )

                env.sim.set_base_pose(
                    "target",
                    final_goal,
                    np.asarray(
                        [
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        dtype=np.float64,
                    ),
                )

                observation = env._get_obs()

                hover_goal = (
                    final_goal.copy()
                )

                hover_goal[2] = HOVER_Z

                initial_cube_z = float(
                    observation[
                        "achieved_goal"
                    ][2]
                )

                max_cube_z = (
                    initial_cube_z
                )

                episode_frame_dir = (
                    staging_dir
                    / f"seed_{seed:06d}"
                )

                shutil.rmtree(
                    episode_frame_dir,
                    ignore_errors=True,
                )

                episode_frame_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                episode_rows = []
                episode_reward = 0.0
                step_index = 0
                stable_count = 0
                teacher_reached_hover = False
                pre_release_reached = False
                full_success = False
                final_metrics: dict[str, Any] = {}

                phase_steps = {
                    "teacher": 0,
                    "descent": 0,
                    "open": 0,
                    "retreat": 0,
                    "settle": 0,
                }

                def execute_step(
                    action: np.ndarray,
                    phase: str,
                    evaluate_success: bool = False,
                ) -> bool:
                    nonlocal observation
                    nonlocal step_index
                    nonlocal stable_count
                    nonlocal full_success
                    nonlocal final_metrics
                    nonlocal episode_reward

                    action = np.asarray(
                        action,
                        dtype=np.float32,
                    ).reshape(4)

                    state_25 = build_state_25(
                        observation
                    )

                    cube_position = np.asarray(
                        observation[
                            "achieved_goal"
                        ],
                        dtype=np.float64,
                    )

                    distance_to_goal = float(
                        np.linalg.norm(
                            cube_position
                            - final_goal
                        )
                    )

                    frame_rel_path = (
                        Path("frames")
                        / f"episode_{episode:04d}"
                        / f"frame_{step_index:04d}.png"
                    )

                    frame_abs_path = (
                        episode_frame_dir
                        / f"frame_{step_index:04d}.png"
                    )

                    save_rgb_frame(
                        env.render(),
                        frame_abs_path,
                    )

                    next_observation = raw_step(
                        env,
                        action,
                    )

                    terminal_success = False

                    if evaluate_success:
                        valid, final_metrics = (
                            full_success_metrics(
                                env,
                                next_observation,
                                final_goal,
                            )
                        )

                        if valid:
                            stable_count += 1
                        else:
                            stable_count = 0

                        terminal_success = (
                            stable_count
                            >= REQUIRED_STABLE_STEPS
                        )

                        if terminal_success:
                            full_success = True

                    reward = (
                        0.0
                        if terminal_success
                        else -1.0
                    )

                    row = {
                        "episode": episode,
                        "seed": seed,
                        "step": step_index,
                        "phase": phase,
                        "task": TASK_TEXT,
                        "frame_path": str(
                            frame_rel_path
                        ),
                        "state_25": json.dumps(
                            state_25.tolist()
                        ),
                        "env_action_4d": json.dumps(
                            action.tolist()
                        ),
                        "smolvla_action_6d": json.dumps(
                            action_4d_to_smolvla_6d(
                                action
                            ).tolist()
                        ),
                        "reward": reward,
                        "done": terminal_success,
                        "is_success": terminal_success,
                        "distance_to_goal": (
                            distance_to_goal
                        ),
                    }

                    episode_rows.append(row)

                    observation = (
                        next_observation
                    )

                    phase_steps[phase] += 1

                    episode_reward += reward
                    step_index += 1

                    return terminal_success

                for _ in range(
                    TEACHER_MAX_STEPS
                ):
                    action = predict_teacher(
                        model,
                        normalizer,
                        observation,
                        hover_goal,
                    )

                    execute_step(
                        action,
                        phase="teacher",
                    )

                    cube_position = np.asarray(
                        observation[
                            "achieved_goal"
                        ],
                        dtype=np.float64,
                    )

                    max_cube_z = max(
                        max_cube_z,
                        float(
                            cube_position[2]
                        ),
                    )

                    hover_distance = float(
                        np.linalg.norm(
                            cube_position
                            - hover_goal
                        )
                    )

                    if (
                        hover_distance <= 0.05
                        and cube_position[2]
                        >= 0.07
                    ):
                        teacher_reached_hover = True
                        break

                if not teacher_reached_hover:
                    rejected_attempts.append(
                        {
                            "seed": seed,
                            "reason": "teacher_hover_failure",
                            "attempted_steps": len(episode_rows),
                            "phase_steps": phase_steps.copy(),
                        }
                    )

                    shutil.rmtree(
                        episode_frame_dir,
                        ignore_errors=True,
                    )

                    print(
                        f"REJECT seed={seed} | "
                        "reason=teacher_hover_failure"
                    )

                    continue

                for _ in range(
                    DESCENT_MAX_STEPS
                ):
                    cube_position = np.asarray(
                        observation[
                            "achieved_goal"
                        ],
                        dtype=np.float64,
                    )

                    displacement = (
                        final_goal
                        - cube_position
                    )

                    displacement[2] = min(
                        float(
                            displacement[2]
                        ),
                        -0.01,
                    )

                    action = cartesian_action(
                        displacement,
                        gripper_command=-1.0,
                    )

                    execute_step(
                        action,
                        phase="descent",
                    )

                    cube_position = np.asarray(
                        observation[
                            "achieved_goal"
                        ],
                        dtype=np.float64,
                    )

                    xy_distance = float(
                        np.linalg.norm(
                            cube_position[:2]
                            - final_goal[:2]
                        )
                    )

                    if (
                        xy_distance
                        <= PRE_RELEASE_XY_TOLERANCE
                        and cube_position[2]
                        <= OBJECT_Z
                        + FINAL_Z_TOLERANCE
                    ):
                        pre_release_reached = True
                        break

                if not pre_release_reached:
                    rejected_attempts.append(
                        {
                            "seed": seed,
                            "reason": "pre_release_failure",
                            "attempted_steps": len(episode_rows),
                            "phase_steps": phase_steps.copy(),
                        }
                    )

                    shutil.rmtree(
                        episode_frame_dir,
                        ignore_errors=True,
                    )

                    print(
                        f"REJECT seed={seed} | "
                        "reason=pre_release_failure"
                    )

                    continue

                for _ in range(
                    OPEN_STEPS
                ):
                    execute_step(
                        np.asarray(
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                            ],
                            dtype=np.float32,
                        ),
                        phase="open",
                    )

                for _ in range(
                    RETREAT_STEPS
                ):
                    execute_step(
                        np.asarray(
                            [
                                0.0,
                                0.0,
                                0.6,
                                1.0,
                            ],
                            dtype=np.float32,
                        ),
                        phase="retreat",
                    )

                for _ in range(
                    SETTLE_MAX_STEPS
                ):
                    terminal = execute_step(
                        np.asarray(
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                            ],
                            dtype=np.float32,
                        ),
                        phase="settle",
                        evaluate_success=True,
                    )

                    if terminal:
                        break

                if not full_success:
                    rejected_attempts.append(
                        {
                            "seed": seed,
                            "reason": "full_release_failure",
                            "attempted_steps": len(episode_rows),
                            "phase_steps": phase_steps.copy(),
                            "stable_count": stable_count,
                            "final_metrics": final_metrics,
                        }
                    )

                    shutil.rmtree(
                        episode_frame_dir,
                        ignore_errors=True,
                    )

                    print(
                        f"REJECT seed={seed} | "
                        "reason=full_release_failure"
                    )

                    continue

                final_episode_frame_dir = (
                    frames_dir
                    / f"episode_{episode:04d}"
                )

                if final_episode_frame_dir.exists():
                    raise FileExistsError(
                        "Accepted episode frame directory "
                        f"already exists: {final_episode_frame_dir}"
                    )

                episode_frame_dir.rename(
                    final_episode_frame_dir
                )

                all_rows.extend(episode_rows)
                total_steps += len(episode_rows)

                for phase, count in phase_steps.items():
                    phase_frame_counts[phase] += count

                final_cube_position = np.asarray(
                    observation[
                        "achieved_goal"
                    ],
                    dtype=np.float64,
                )

                final_distance = float(
                    np.linalg.norm(
                        final_cube_position
                        - final_goal
                    )
                )

                episode_summary = {
                    "episode": episode,
                    "seed": seed,
                    "success": full_success,
                    "steps": len(
                        episode_rows
                    ),
                    "reward": episode_reward,
                    "teacher_reached_hover": (
                        teacher_reached_hover
                    ),
                    "pre_release_reached": (
                        pre_release_reached
                    ),
                    "stable_count": (
                        stable_count
                    ),
                    "lift_height": (
                        max_cube_z
                        - initial_cube_z
                    ),
                    "final_distance": (
                        final_distance
                    ),
                    "phase_steps": (
                        phase_steps
                    ),
                    "final_metrics": (
                        final_metrics
                    ),
                }

                episode_summaries.append(
                    episode_summary
                )

                write_json(
                    episodes_dir
                    / f"episode_{episode:04d}.json",
                    {
                        "episode_summary": (
                            episode_summary
                        ),
                        "steps": episode_rows,
                    },
                )

                print(
                    f"EP {episode:04d} | "
                    f"seed={seed} | "
                    f"success={full_success} | "
                    f"steps={len(episode_rows)} | "
                    f"lift={episode_summary['lift_height']:.4f} | "
                    f"xy={final_metrics['xy_distance']:.4f}"
                )

                episode += 1

            if episode < args.episodes:
                raise RuntimeError(
                    "Could not collect the requested number "
                    "of successful episodes within --max-attempts"
                )

            writer.writerows(all_rows)

        metadata = {
            "env_id": "PandaPickAndPlace-v3",
            "dataset_variant": (
                "full_table_release"
            ),
            "teacher": (
                "chencliu/"
                "tqc-PandaPickAndPlace-v3 "
                "+ Cartesian release controller"
            ),
            "model_path": str(model_path),
            "vec_normalize_path": str(
                vec_path
            ),
            "task": TASK_TEXT,
            "state_shape": [25],
            "env_action_shape": [4],
            "smolvla_action_shape": [6],
            "image_source": "env.render()",
            "image_format": "png",
            "requested_episodes": args.episodes,
            "episodes": len(episode_summaries),
            "seed_start": args.seed_start,
            "max_attempts": max_attempts,
            "attempted_seed_count": attempted_seed_count,
            "accepted_seeds": [
                episode["seed"]
                for episode in episode_summaries
            ],
            "rejected_seed_count": len(rejected_attempts),
            "rejected_attempts": rejected_attempts,
            "export_complete": True,
            "total_steps": total_steps,
            "phase_frame_counts": (
                phase_frame_counts
            ),
            "hover_z": HOVER_Z,
            "pre_release_xy_tolerance": (
                PRE_RELEASE_XY_TOLERANCE
            ),
            "final_xy_tolerance": (
                FINAL_XY_TOLERANCE
            ),
            "required_stable_steps": (
                REQUIRED_STABLE_STEPS
            ),
            "settle_max_steps": (
                SETTLE_MAX_STEPS
            ),
        }

        success_values = [
            episode["success"]
            for episode in episode_summaries
        ]

        summary = {
            **metadata,
            "success_count": int(
                np.sum(success_values)
            ),
            "success_rate": float(
                np.mean(success_values)
            ),
            "mean_steps": float(
                np.mean(
                    [
                        episode["steps"]
                        for episode
                        in episode_summaries
                    ]
                )
            ),
            "mean_lift_height": float(
                np.mean(
                    [
                        episode["lift_height"]
                        for episode
                        in episode_summaries
                    ]
                )
            ),
            "mean_final_distance": float(
                np.mean(
                    [
                        episode[
                            "final_distance"
                        ]
                        for episode
                        in episode_summaries
                    ]
                )
            ),
            "episode_summaries": (
                episode_summaries
            ),
        }

        write_json(
            args.out_dir / "metadata.json",
            metadata,
        )

        write_json(
            args.out_dir / "summary.json",
            summary,
        )

        print("\nEXPORT_OK")
        print(
            f"Output directory: "
            f"{args.out_dir}"
        )
        print(f"CSV: {csv_path}")
        print(
            f"Total frames: "
            f"{total_steps}"
        )
        print(
            f"Success: "
            f"{summary['success_count']}"
            f"/{args.episodes}"
        )
        print(
            f"Attempted seeds: "
            f"{attempted_seed_count}"
        )
        print(
            f"Rejected seeds: "
            f"{len(rejected_attempts)}"
        )
        print(
            "Phase frames: "
            f"{phase_frame_counts}"
        )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )
        env.close()
        normalization_env.close()


if __name__ == "__main__":
    main()
