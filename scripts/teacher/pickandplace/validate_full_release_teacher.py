from __future__ import annotations

import json
import shutil
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import panda_gym  # noqa: F401

from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SEEDS = list(range(100))

MODEL_DIR = Path(
    "models/PandaPickAndPlace/"
    "chencliu_tqc_PandaPickAndPlace_v3"
)

OUT_DIR = Path(
    "outputs/teacher_diagnostics/"
    "panda_pickandplace_full_release_validation_100_release_xy_015"
)

OBJECT_Z = 0.02
HOVER_Z = 0.12
PRE_RELEASE_XY_TOLERANCE = 0.015

TEACHER_MAX_STEPS = 35
DESCENT_MAX_STEPS = 20
OPEN_STEPS = 4
RETREAT_STEPS = 6
SETTLE_STEPS = 15
REQUIRED_STABLE_STEPS = 5

FPS = 10
SAVE_VIDEOS = False


def record_frame(writer, env) -> None:
    """Render and save a frame only when video recording is enabled."""
    if writer is not None:
        writer.append_data(env.render())


def make_base_env():
    return gym.make(
        "PandaPickAndPlace-v3",
        render_mode="rgb_array",
        renderer="Tiny",
    )


def raw_step(env, action):
    """Execute one simulation step without early task termination."""
    env.robot.set_action(
        np.asarray(action, dtype=np.float32)
    )
    env.sim.step()
    return env._get_obs()


def predict_teacher(
    model,
    normalizer,
    observation,
    proxy_goal,
):
    """Predict a teacher action using an elevated proxy goal."""
    batch = {
        key: np.expand_dims(
            np.asarray(value),
            axis=0,
        )
        for key, value in observation.items()
    }

    batch["desired_goal"] = np.expand_dims(
        np.asarray(proxy_goal, dtype=np.float32),
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
    displacement,
    gripper_command,
):
    """Convert a Cartesian displacement into Panda EE control."""
    xyz_action = np.clip(
        np.asarray(displacement, dtype=np.float32) / 0.05,
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


def contact_flags(env):
    """Read finger-object and object-table contacts."""
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


def success_metrics(
    env,
    observation,
    final_goal,
):
    """Evaluate full placement, release and stability."""
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
            cube_position[:2] - final_goal[:2]
        )
    )

    z_error = abs(
        float(cube_position[2]) - OBJECT_Z
    )

    linear_speed = float(
        np.linalg.norm(cube_velocity)
    )

    angular_speed = float(
        np.linalg.norm(cube_angular_velocity)
    )

    checks = {
        "xy_close": xy_distance <= 0.04,
        "on_table": z_error <= 0.012,
        "gripper_open": fingers_width >= 0.06,
        "released": (
            not contacts["left"]
            and not contacts["right"]
        ),
        "table_contact": contacts["table"],
        "stable_linear": linear_speed <= 0.025,
        "stable_angular": angular_speed <= 0.5,
    }

    metrics = {
        "cube_position": cube_position.tolist(),
        "xy_distance": xy_distance,
        "z_error": z_error,
        "fingers_width": fingers_width,
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
        "contacts": contacts,
        "checks": checks,
    }

    return all(checks.values()), metrics


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalization_env = DummyVecEnv(
        [make_base_env]
    )

    normalizer = VecNormalize.load(
        str(MODEL_DIR / "vec_normalize.pkl"),
        normalization_env,
    )

    normalizer.training = False
    normalizer.norm_reward = False

    model = TQC.load(
        str(
            MODEL_DIR
            / "tqc-PandaPickAndPlace-v3.zip"
        ),
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

    results = []

    try:
        for seed in SEEDS:
            observation, _ = env.reset(seed=seed)

            final_goal = np.asarray(
                observation["desired_goal"],
                dtype=np.float64,
            )

            # Force the real destination to be on the table.
            final_goal[2] = OBJECT_Z

            env.task.goal = final_goal.copy()

            env.sim.set_base_pose(
                "target",
                final_goal,
                np.asarray(
                    [0.0, 0.0, 0.0, 1.0]
                ),
            )

            observation = env._get_obs()

            # The RL teacher receives a temporary elevated goal.
            hover_goal = final_goal.copy()
            hover_goal[2] = HOVER_Z

            initial_cube_z = float(
                observation["achieved_goal"][2]
            )

            max_cube_z = initial_cube_z
            teacher_reached_hover = False
            stable_count = 0
            final_metrics = {}

            phase_steps = {
                "teacher": 0,
                "descent": 0,
                "open": 0,
                "retreat": 0,
                "settle": 0,
            }

            video_path = (
                OUT_DIR
                / f"seed_{seed:04d}.mp4"
            )

            writer = (
                imageio.get_writer(
                    video_path,
                    fps=FPS,
                    codec="libx264",
                    quality=8,
                )
                if SAVE_VIDEOS
                else None
            )

            try:
                record_frame(writer, env)

                # Phase 1: teacher grasp, lift and transport.
                for _ in range(
                    TEACHER_MAX_STEPS
                ):
                    action = predict_teacher(
                        model,
                        normalizer,
                        observation,
                        hover_goal,
                    )

                    observation = raw_step(
                        env,
                        action,
                    )

                    record_frame(writer, env)

                    phase_steps["teacher"] += 1

                    cube_position = np.asarray(
                        observation[
                            "achieved_goal"
                        ],
                        dtype=np.float64,
                    )

                    max_cube_z = max(
                        max_cube_z,
                        float(cube_position[2]),
                    )

                    hover_distance = float(
                        np.linalg.norm(
                            cube_position
                            - hover_goal
                        )
                    )

                    if (
                        hover_distance <= 0.05
                        and cube_position[2] >= 0.07
                    ):
                        teacher_reached_hover = True
                        break

                if teacher_reached_hover:
                    # Phase 2: closed-loop descent.
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

                        # Always preserve a downward command
                        # until the cube reaches the table.
                        displacement[2] = min(
                            float(displacement[2]),
                            -0.01,
                        )

                        action = cartesian_action(
                            displacement,
                            gripper_command=-1.0,
                        )

                        observation = raw_step(
                            env,
                            action,
                        )

                        record_frame(writer, env)

                        phase_steps[
                            "descent"
                        ] += 1

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
                            <= OBJECT_Z + 0.012
                        ):
                            break

                    # Phase 3: open the gripper.
                    for _ in range(OPEN_STEPS):
                        observation = raw_step(
                            env,
                            [0.0, 0.0, 0.0, 1.0],
                        )

                        record_frame(writer, env)

                        phase_steps["open"] += 1

                    # Phase 4: retreat vertically.
                    for _ in range(
                        RETREAT_STEPS
                    ):
                        observation = raw_step(
                            env,
                            [0.0, 0.0, 0.6, 1.0],
                        )

                        record_frame(writer, env)

                        phase_steps[
                            "retreat"
                        ] += 1

                    # Phase 5: wait for cube stability.
                    for _ in range(
                        SETTLE_STEPS
                    ):
                        observation = raw_step(
                            env,
                            [0.0, 0.0, 0.0, 1.0],
                        )

                        record_frame(writer, env)

                        phase_steps[
                            "settle"
                        ] += 1

                        valid, final_metrics = (
                            success_metrics(
                                env,
                                observation,
                                final_goal,
                            )
                        )

                        if valid:
                            stable_count += 1
                        else:
                            stable_count = 0

                # Hold the final frame for inspection.
                for _ in range(FPS):
                    record_frame(writer, env)

            finally:
                if writer is not None:
                    writer.close()

            full_success = (
                stable_count
                >= REQUIRED_STABLE_STEPS
            )

            result = {
                "seed": seed,
                "teacher_reached_hover": (
                    teacher_reached_hover
                ),
                "full_success": full_success,
                "stable_count": stable_count,
                "lift_height": (
                    max_cube_z
                    - initial_cube_z
                ),
                "phase_steps": phase_steps,
                "final_goal": (
                    final_goal.tolist()
                ),
                "final_metrics": final_metrics,
                "video": (
                    str(video_path)
                    if SAVE_VIDEOS
                    else None
                ),
            }

            results.append(result)

            print(
                f"seed={seed:03d} "
                f"hover={teacher_reached_hover} "
                f"full_success={full_success} "
                f"stable={stable_count} "
                f"lift={result['lift_height']:.4f} "
                f"xy={final_metrics.get('xy_distance')}"
            )

        summary = {
            "episode_count": len(results),
            "teacher_hover_success_count": sum(
                result[
                    "teacher_reached_hover"
                ]
                for result in results
            ),
            "full_success_count": sum(
                result["full_success"]
                for result in results
            ),
            "results": results,
        }

        summary["teacher_hover_success_rate"] = (
            summary["teacher_hover_success_count"]
            / summary["episode_count"]
        )

        summary["full_success_rate"] = (
            summary["full_success_count"]
            / summary["episode_count"]
        )

        summary["hover_failure_seeds"] = [
            result["seed"]
            for result in results
            if not result["teacher_reached_hover"]
        ]

        summary["full_failure_seeds"] = [
            result["seed"]
            for result in results
            if not result["full_success"]
        ]

        summary["save_videos"] = SAVE_VIDEOS

        summary_path = (
            OUT_DIR / "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            "\nFULL_RELEASE_VALIDATION"
        )

        print(
            json.dumps(
                {
                    "episode_count": (
                        summary[
                            "episode_count"
                        ]
                    ),
                    "teacher_hover_success_count": (
                        summary[
                            "teacher_hover_success_count"
                        ]
                    ),
                    "full_success_count": (
                        summary[
                            "full_success_count"
                        ]
                    ),
                },
                indent=2,
            )
        )

        print(f"saved={summary_path}")

    finally:
        env.close()
        normalization_env.close()


if __name__ == "__main__":
    main()
