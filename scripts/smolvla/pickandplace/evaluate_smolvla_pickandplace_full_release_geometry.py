from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv
from scripts.smolvla.pickandplace import (
    evaluate_smolvla_pickandplace_rollout as base_eval,
)
from scripts.teacher.pickandplace.validate_full_release_teacher import (
    OBJECT_Z,
    REQUIRED_STABLE_STEPS,
    raw_step,
    success_metrics,
)


DEFAULT_TASK_TEXT = (
    "pick up the cube, place it on the target, release it, "
    "and move the gripper away"
)

ORIGINAL_SUCCESS_DISTANCE = 0.05
MIN_LIFT_HEIGHT = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-repo-id",
        required=True,
    )
    parser.add_argument(
        "--task-text",
        default=DEFAULT_TASK_TEXT,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=75,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {args.dataset_root}"
        )

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be positive"
        )

    if args.max_steps <= 0:
        raise ValueError(
            "--max-steps must be positive"
        )

    if args.seed < 0:
        raise ValueError(
            "--seed must be non-negative"
        )

    if args.action_scale <= 0.0:
        raise ValueError(
            "--action-scale must be positive"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    base_eval.seed_episode(args.seed)
    base_eval.TASK_TEXT = args.task_text

    dataset_stats = base_eval.get_dataset_stats(
        args.dataset_repo_id,
        args.dataset_root,
    )

    policy = (
        SmolVLAPolicy.from_pretrained(
            str(args.checkpoint)
        )
        .to(device)
        .eval()
    )

    policy.config.device = device

    preprocessor, postprocessor = (
        make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_stats,
        )
    )

    env = PandaPickAndPlaceEnv(
        render_mode="rgb_array",
        renderer="Tiny",
    )

    results = []

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode

            base_eval.seed_episode(episode_seed)
            base_eval.reset_policy_if_supported(
                policy
            )

            observation, _ = env.reset(
                seed=episode_seed
            )

            final_goal = np.asarray(
                observation["desired_goal"],
                dtype=np.float64,
            )

            final_goal[2] = OBJECT_Z
            env.task.goal = final_goal.copy()

            env.sim.set_base_pose(
                "target",
                final_goal,
                np.asarray(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
            )

            observation = env._get_obs()

            initial_cube_position = np.asarray(
                observation["achieved_goal"],
                dtype=np.float64,
            )

            initial_target_position = np.asarray(
                final_goal,
                dtype=np.float64,
            )

            initial_delta_xyz = (
                initial_cube_position
                - initial_target_position
            )

            initial_xy_distance = float(
                np.linalg.norm(
                    initial_delta_xyz[:2]
                )
            )

            initial_goal_distance_3d = float(
                np.linalg.norm(
                    initial_delta_xyz
                )
            )

            initial_state_vector = np.asarray(
                observation.get(
                    "observation",
                    [],
                ),
                dtype=np.float64,
            )

            initial_cube_z = float(
                observation["achieved_goal"][2]
            )

            max_cube_z = initial_cube_z

            initial_distance = float(
                np.linalg.norm(
                    np.asarray(
                        observation[
                            "achieved_goal"
                        ],
                        dtype=np.float64,
                    )
                    - final_goal
                )
            )

            initial_original_success = (
                initial_distance
                < ORIGINAL_SUCCESS_DISTANCE
            )

            best_distance = initial_distance
            stable_count = 0
            max_stable_count = 0
            lift_achieved_ever = False
            original_success_ever = (
                initial_original_success
            )
            nontrivial_original_success_ever = False
            full_success = False
            final_metrics = {}
            ever_checks = Counter()
            post_lift_ever_checks = Counter()
            steps = 0

            for step in range(args.max_steps):
                frame = env.render()

                action_6d = (
                    base_eval.select_smolvla_action(
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        obs=observation,
                        frame=frame,
                        device=device,
                    )
                )

                action_4d = np.clip(
                    action_6d[:4]
                    * args.action_scale,
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)

                observation = raw_step(
                    env,
                    action_4d,
                )

                cube_position = np.asarray(
                    observation["achieved_goal"],
                    dtype=np.float64,
                )

                max_cube_z = max(
                    max_cube_z,
                    float(cube_position[2]),
                )

                current_lift_height = (
                    max_cube_z
                    - initial_cube_z
                )

                if (
                    current_lift_height
                    >= MIN_LIFT_HEIGHT
                ):
                    lift_achieved_ever = True

                distance = float(
                    np.linalg.norm(
                        cube_position - final_goal
                    )
                )

                best_distance = min(
                    best_distance,
                    distance,
                )

                current_original_success = (
                    distance
                    < ORIGINAL_SUCCESS_DISTANCE
                )

                original_success_ever |= (
                    current_original_success
                )

                if (
                    current_original_success
                    and not initial_original_success
                ):
                    nontrivial_original_success_ever = True

                valid, final_metrics = (
                    success_metrics(
                        env,
                        observation,
                        final_goal,
                    )
                )

                for (
                    check_name,
                    check_value,
                ) in final_metrics[
                    "checks"
                ].items():
                    if check_value:
                        ever_checks[
                            check_name
                        ] = 1

                        if lift_achieved_ever:
                            post_lift_ever_checks[
                                check_name
                            ] = 1

                if (
                    valid
                    and lift_achieved_ever
                ):
                    stable_count += 1
                else:
                    stable_count = 0

                max_stable_count = max(
                    max_stable_count,
                    stable_count,
                )

                steps = step + 1

                if (
                    stable_count
                    >= REQUIRED_STABLE_STEPS
                ):
                    full_success = True
                    break

            result = {
                "episode": episode,
                "seed": episode_seed,
                "initial_geometry": {
                    "cube_position": (
                        initial_cube_position.tolist()
                    ),
                    "target_position": (
                        initial_target_position.tolist()
                    ),
                    "delta_xyz": (
                        initial_delta_xyz.tolist()
                    ),
                    "xy_distance": (
                        initial_xy_distance
                    ),
                    "distance_3d": (
                        initial_goal_distance_3d
                    ),
                    "state_vector": (
                        initial_state_vector.tolist()
                    ),
                },
                "full_success": full_success,
                "initial_original_success": (
                    initial_original_success
                ),
                "original_success_ever": (
                    original_success_ever
                ),
                "nontrivial_original_success_ever": (
                    nontrivial_original_success_ever
                ),
                "steps": steps,
                "max_stable_count": (
                    max_stable_count
                ),
                "minimum_lift_height": (
                    MIN_LIFT_HEIGHT
                ),
                "lift_achieved": (
                    lift_achieved_ever
                ),
                "lift_height": (
                    max_cube_z
                    - initial_cube_z
                ),
                "initial_goal_distance": (
                    initial_distance
                ),
                "best_goal_distance": (
                    best_distance
                ),
                "final_metrics": (
                    final_metrics
                ),
                "ever_checks": {
                    check_name: bool(
                        ever_checks[
                            check_name
                        ]
                    )
                    for check_name in sorted(
                        final_metrics.get(
                            "checks",
                            {},
                        )
                    )
                },
                "post_lift_ever_checks": {
                    check_name: bool(
                        post_lift_ever_checks[
                            check_name
                        ]
                    )
                    for check_name in sorted(
                        final_metrics.get(
                            "checks",
                            {},
                        )
                    )
                },
            }

            results.append(result)

            print(
                f"EP {episode:04d} | "
                f"seed={episode_seed} | "
                f"full_success={full_success} | "
                f"original_success="
                f"{original_success_ever} | "
                f"initial_trivial="
                f"{initial_original_success} | "
                f"nontrivial_success="
                f"{nontrivial_original_success_ever} | "
                f"steps={steps} | "
                f"stable={max_stable_count} | "
                f"lift_ok={lift_achieved_ever} | "
                f"lift="
                f"{result['lift_height']:.4f} | "
                f"xy="
                f"{final_metrics.get('xy_distance')}"
            )

    finally:
        env.close()

    full_values = [
        result["full_success"]
        for result in results
    ]

    original_values = [
        result["original_success_ever"]
        for result in results
    ]

    lift_achieved_values = [
        result["lift_achieved"]
        for result in results
    ]

    initial_original_values = [
        result["initial_original_success"]
        for result in results
    ]

    nontrivial_original_values = [
        result[
            "nontrivial_original_success_ever"
        ]
        for result in results
    ]

    nontrivial_eligible_values = [
        not result["initial_original_success"]
        for result in results
    ]

    nontrivial_eligible_count = int(
        np.sum(nontrivial_eligible_values)
    )

    nontrivial_success_count = int(
        np.sum(nontrivial_original_values)
    )

    check_names = sorted(
        {
            check_name
            for result in results
            for check_name in (
                result["final_metrics"]
                .get("checks", {})
            )
        }
    )

    summary = {
        "env_id": "PandaPickAndPlace-v3",
        "evaluation_mode": "full_release",
        "checkpoint": str(args.checkpoint),
        "dataset_repo_id": (
            args.dataset_repo_id
        ),
        "dataset_root": str(
            args.dataset_root
        ),
        "task_text": args.task_text,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed_start": args.seed,
        "seed_end": (
            args.seed
            + args.episodes
            - 1
        ),
        "action_scale": args.action_scale,
        "device": device,
        "required_stable_steps": (
            REQUIRED_STABLE_STEPS
        ),
        "minimum_lift_height": (
            MIN_LIFT_HEIGHT
        ),
        "success_requires_prior_lift": True,
        "records_initial_geometry": True,
        "initial_geometry_fields": [
            "cube_position",
            "target_position",
            "delta_xyz",
            "xy_distance",
            "distance_3d",
            "state_vector",
        ],
        "lift_achieved_count": int(
            np.sum(lift_achieved_values)
        ),
        "lift_achieved_rate": float(
            np.mean(lift_achieved_values)
        ),
        "full_success_count": int(
            np.sum(full_values)
        ),
        "full_success_rate": float(
            np.mean(full_values)
        ),
        "original_success_count": int(
            np.sum(original_values)
        ),
        "original_success_rate": float(
            np.mean(original_values)
        ),
        "initial_original_success_count": int(
            np.sum(initial_original_values)
        ),
        "initial_original_success_rate": float(
            np.mean(initial_original_values)
        ),
        "nontrivial_eligible_count": (
            nontrivial_eligible_count
        ),
        "nontrivial_original_success_count": (
            nontrivial_success_count
        ),
        "nontrivial_original_success_rate": (
            float(
                nontrivial_success_count
                / nontrivial_eligible_count
            )
            if nontrivial_eligible_count > 0
            else None
        ),
        "ever_check_counts": {
            check_name: int(
                sum(
                    result[
                        "ever_checks"
                    ].get(
                        check_name,
                        False,
                    )
                    for result in results
                )
            )
            for check_name in check_names
        },
        "final_check_counts": {
            check_name: int(
                sum(
                    result[
                        "final_metrics"
                    ]
                    .get("checks", {})
                    .get(
                        check_name,
                        False,
                    )
                    for result in results
                )
            )
            for check_name in check_names
        },
        "post_lift_ever_check_counts": {
            check_name: int(
                sum(
                    result[
                        "post_lift_ever_checks"
                    ].get(
                        check_name,
                        False,
                    )
                    for result in results
                )
            )
            for check_name in check_names
        },
        "results": results,
    }

    summary_path = (
        args.out_dir
        / "full_release_rollout_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nFULL_RELEASE_ROLLOUT_SUMMARY"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"\nsaved_summary={summary_path}"
    )


if __name__ == "__main__":
    main()
