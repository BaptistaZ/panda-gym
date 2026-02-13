import argparse
import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import panda_gym  # noqa: F401  (regista envs)


def record_episode(
    env_id: str,
    out_path: str,
    policy: str,
    max_steps: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
):
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        renderer="Tiny",
    )
    obs, _ = env.reset(seed=seed)
    robot = env.unwrapped.robot

    has_gripper = not getattr(robot, "block_gripper", False)
    step_scale = 0.05  # consistente com panda_gym (escala interna)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)

    # Estados internos (máquinas de estados)
    phase_pnp = "OPEN_INIT"
    open_steps = 3
    close_steps = 8
    open_release_steps = 6

    phase_push = "GO_ABOVE_OBJ"

    phase_stack = "OPEN_INIT"
    stack_subtask = 1  # 1=colocar obj1 no goal1, 2=colocar obj2 no goal2
    stack_open_steps = 3
    stack_close_steps = 8
    stack_release_steps = 6

    phase_flip = "GO_ABOVE_OBJ"
    flip_close_steps = 10
    flip_shake_steps = 40
    flip_release_steps = 10

    def arm_q():
        return np.array([robot.get_joint_angle(joint=i) for i in range(7)], dtype=np.float64)

    def ee_move_action(target_pos: np.ndarray):
        ee = robot.get_ee_position()
        return np.clip((target_pos - ee) / step_scale, -1, 1).astype(np.float32)

    def joints_move_action(target_pos: np.ndarray):
        current_q = arm_q()
        ik = np.array(
            robot.inverse_kinematics(
                link=robot.ee_link,
                position=target_pos,
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            dtype=np.float64,
        )[:7]
        return np.clip((ik - current_q) / step_scale, -1, 1).astype(np.float32)

    def make_action(target_pos: np.ndarray, gripper_cmd: float | None = None):
        if robot.control_type == "ee":
            core = ee_move_action(target_pos)
        else:
            core = joints_move_action(target_pos)

        if not has_gripper:
            # Push/Reach/Slide: ação é só core (3 ou 7)
            return core

        # Env com garra (PickAndPlace/Stack/Flip): core + 1 dimensão (garra)
        g = 0.0 if gripper_cmd is None else float(gripper_cmd)
        a = np.zeros(core.shape[0] + 1, dtype=np.float32)
        a[: core.shape[0]] = core
        a[-1] = g
        return a

    def near(a: np.ndarray, b: np.ndarray, thr: float) -> bool:
        return float(np.linalg.norm(a - b)) < thr

    # frame inicial (usa SEMPRE as mesmas dimensões)
    frame = env.unwrapped.robot.sim.render(width=width, height=height)
    writer.append_data(frame)

    try:
        done = False
        n = 0

        while not done and n < max_steps:
            # ---------------------------
            # RANDOM
            # ---------------------------
            if policy == "random":
                action = env.action_space.sample()

            # ---------------------------
            # REACH_GOAL (para Reach)
            # ---------------------------
            elif policy == "reach_goal":
                ag = obs["achieved_goal"]
                dg = obs["desired_goal"]
                if np.shape(dg) != (3,) or np.shape(ag) != (3,):
                    raise ValueError("reach_goal assume achieved_goal/desired_goal com shape (3,)")

                if robot.control_type == "ee":
                    core = np.clip((dg - ag) / step_scale, -1, 1).astype(np.float32)
                    action = core if not has_gripper else np.concatenate([core, np.array([0.0], dtype=np.float32)])
                else:
                    target_q = np.array(
                        robot.inverse_kinematics(
                            link=robot.ee_link,
                            position=dg,
                            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                        ),
                        dtype=np.float64,
                    )[:7]
                    core = np.clip((target_q - arm_q()) / step_scale, -1, 1).astype(np.float32)
                    action = core if not has_gripper else np.concatenate([core, np.array([0.0], dtype=np.float32)])

            # ---------------------------
            # PUSH_PHASES / SLIDE_PHASES (sem garra)
            # ---------------------------
            elif policy in ("push_phases", "slide_phases"):
                if has_gripper:
                    raise ValueError(f"{policy} foi pensado para envs sem garra (Push/Slide).")

                obj = obs["achieved_goal"].copy()
                goal = obs["desired_goal"].copy()
                ee = robot.get_ee_position().copy()

                # Direção no plano XY
                d = (goal - obj).copy()
                d[2] = 0.0
                norm = float(np.linalg.norm(d))
                if norm < 1e-9:
                    d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    norm = 1.0
                d /= norm

                above_obj = obj.copy()
                above_obj[2] = obj[2] + 0.12

                contact_z = max(float(obj[2]), 0.02)
                behind = obj.copy()
                behind[:2] = obj[:2] - d[:2] * 0.10
                behind[2] = contact_z

                push_end = goal.copy()
                push_end[2] = contact_z

                if phase_push == "GO_ABOVE_OBJ":
                    action = make_action(above_obj)
                    if near(ee, above_obj, 0.04):
                        phase_push = "GO_BEHIND_OBJ"

                elif phase_push == "GO_BEHIND_OBJ":
                    action = make_action(behind)
                    if near(ee, behind, 0.03):
                        phase_push = "PUSH"

                else:  # "PUSH"
                    action = make_action(push_end)

            # ---------------------------
            # PICK_AND_PLACE_PHASES (com garra)
            # ---------------------------
            elif policy == "pick_and_place_phases":
                if not has_gripper:
                    raise ValueError("pick_and_place_phases requer env com garra (PickAndPlace/Stack/Flip).")

                obj = obs["achieved_goal"].copy()
                goal = obs["desired_goal"].copy()
                ee = robot.get_ee_position().copy()

                above_obj = obj.copy()
                above_obj[2] = obj[2] + 0.12
                grasp_pos = obj.copy()
                grasp_pos[2] = obj[2] + 0.02
                lift_pos = obj.copy()
                lift_pos[2] = obj[2] + 0.15
                above_goal = goal.copy()
                above_goal[2] = goal[2] + 0.12
                place_pos = goal.copy()
                place_pos[2] = goal[2] + 0.03

                if phase_pnp == "OPEN_INIT":
                    action = make_action(above_obj, gripper_cmd=+1)
                    open_steps -= 1
                    if open_steps <= 0:
                        phase_pnp = "GO_ABOVE_OBJ"

                elif phase_pnp == "GO_ABOVE_OBJ":
                    action = make_action(above_obj, gripper_cmd=+1)
                    if near(ee, above_obj, 0.03):
                        phase_pnp = "DESCEND"

                elif phase_pnp == "DESCEND":
                    action = make_action(grasp_pos, gripper_cmd=+1)
                    if near(ee, grasp_pos, 0.02):
                        phase_pnp = "CLOSE"

                elif phase_pnp == "CLOSE":
                    action = make_action(grasp_pos, gripper_cmd=-1)
                    close_steps -= 1
                    if close_steps <= 0:
                        phase_pnp = "LIFT"

                elif phase_pnp == "LIFT":
                    action = make_action(lift_pos, gripper_cmd=-1)
                    if n > 20:
                        phase_pnp = "GO_ABOVE_GOAL"

                elif phase_pnp == "GO_ABOVE_GOAL":
                    action = make_action(above_goal, gripper_cmd=-1)
                    if near(ee, above_goal, 0.04):
                        phase_pnp = "PLACE_DESCEND"

                elif phase_pnp == "PLACE_DESCEND":
                    action = make_action(place_pos, gripper_cmd=-1)
                    if near(ee, place_pos, 0.03):
                        phase_pnp = "RELEASE"

                elif phase_pnp == "RELEASE":
                    action = make_action(place_pos, gripper_cmd=+1)
                    open_release_steps -= 1
                    if open_release_steps <= 0:
                        phase_pnp = "RETREAT"

                else:  # RETREAT/DEFAULT
                    action = make_action(above_goal, gripper_cmd=+1)

            # ---------------------------
            # STACK_PHASES (com garra, 2 objetos)
            # ---------------------------
            elif policy == "stack_phases":
                if not has_gripper:
                    raise ValueError("stack_phases requer env com garra (PandaStack*).")

                ag = obs["achieved_goal"].copy()
                dg = obs["desired_goal"].copy()
                if np.shape(ag) != (6,) or np.shape(dg) != (6,):
                    raise ValueError("stack_phases assume achieved_goal/desired_goal com shape (6,)")

                obj1 = ag[:3].copy()
                obj2 = ag[3:6].copy()
                goal1 = dg[:3].copy()
                goal2 = dg[3:6].copy()
                ee = robot.get_ee_position().copy()

                if stack_subtask == 1:
                    obj = obj1
                    goal = goal1
                else:
                    obj = obj2
                    goal = goal2

                above_obj = obj.copy()
                above_obj[2] = obj[2] + 0.12
                grasp_pos = obj.copy()
                grasp_pos[2] = obj[2] + 0.02
                lift_pos = obj.copy()
                lift_pos[2] = obj[2] + 0.18
                above_goal = goal.copy()
                above_goal[2] = goal[2] + 0.12
                place_pos = goal.copy()
                place_pos[2] = goal[2] + 0.03

                if phase_stack == "OPEN_INIT":
                    action = make_action(above_obj, gripper_cmd=+1)
                    stack_open_steps -= 1
                    if stack_open_steps <= 0:
                        phase_stack = "GO_ABOVE_OBJ"

                elif phase_stack == "GO_ABOVE_OBJ":
                    action = make_action(above_obj, gripper_cmd=+1)
                    if near(ee, above_obj, 0.04):
                        phase_stack = "DESCEND"

                elif phase_stack == "DESCEND":
                    action = make_action(grasp_pos, gripper_cmd=+1)
                    if near(ee, grasp_pos, 0.025):
                        phase_stack = "CLOSE"

                elif phase_stack == "CLOSE":
                    action = make_action(grasp_pos, gripper_cmd=-1)
                    stack_close_steps -= 1
                    if stack_close_steps <= 0:
                        phase_stack = "LIFT"

                elif phase_stack == "LIFT":
                    action = make_action(lift_pos, gripper_cmd=-1)
                    if n > 25:
                        phase_stack = "GO_ABOVE_GOAL"

                elif phase_stack == "GO_ABOVE_GOAL":
                    action = make_action(above_goal, gripper_cmd=-1)
                    if near(ee, above_goal, 0.05):
                        phase_stack = "PLACE_DESCEND"

                elif phase_stack == "PLACE_DESCEND":
                    action = make_action(place_pos, gripper_cmd=-1)
                    if near(ee, place_pos, 0.04):
                        phase_stack = "RELEASE"

                elif phase_stack == "RELEASE":
                    action = make_action(place_pos, gripper_cmd=+1)
                    stack_release_steps -= 1
                    if stack_release_steps <= 0:
                        phase_stack = "RETREAT"

                else:  # RETREAT
                    action = make_action(above_goal, gripper_cmd=+1)
                    # passar para o 2º objeto quando termina o 1º
                    if stack_subtask == 1:
                        stack_subtask = 2
                        phase_stack = "OPEN_INIT"
                        stack_open_steps = 3
                        stack_close_steps = 8
                        stack_release_steps = 6

            # ---------------------------
            # FLIP_PHASES (heurística para induzir rotação)
            # ---------------------------
            elif policy == "flip_phases":
                if not has_gripper:
                    raise ValueError("flip_phases requer env com garra (PandaFlip*).")

                ee = robot.get_ee_position().copy()
                obj_pos = np.array(obs["observation"][:3], dtype=np.float64)

                above = obj_pos.copy()
                above[2] = obj_pos[2] + 0.14
                grasp = obj_pos.copy()
                grasp[2] = obj_pos[2] + 0.03
                lift = obj_pos.copy()
                lift[2] = obj_pos[2] + 0.20

                # “shake” lateral para tentar induzir rotação
                shake_left = lift.copy()
                shake_left[1] -= 0.08
                shake_right = lift.copy()
                shake_right[1] += 0.08

                if phase_flip == "GO_ABOVE_OBJ":
                    action = make_action(above, gripper_cmd=+1)
                    if near(ee, above, 0.05):
                        phase_flip = "DESCEND"

                elif phase_flip == "DESCEND":
                    action = make_action(grasp, gripper_cmd=+1)
                    if near(ee, grasp, 0.04):
                        phase_flip = "CLOSE"

                elif phase_flip == "CLOSE":
                    action = make_action(grasp, gripper_cmd=-1)
                    flip_close_steps -= 1
                    if flip_close_steps <= 0:
                        phase_flip = "LIFT"

                elif phase_flip == "LIFT":
                    action = make_action(lift, gripper_cmd=-1)
                    if near(ee, lift, 0.06):
                        phase_flip = "SHAKE"

                elif phase_flip == "SHAKE":
                    # alterna entre esquerda/direita
                    target = shake_left if (flip_shake_steps % 2 == 0) else shake_right
                    action = make_action(target, gripper_cmd=-1)
                    flip_shake_steps -= 1
                    if flip_shake_steps <= 0:
                        phase_flip = "RELEASE"

                elif phase_flip == "RELEASE":
                    action = make_action(above, gripper_cmd=+1)
                    flip_release_steps -= 1
                    if flip_release_steps <= 0:
                        phase_flip = "RESET"

                else:  # RESET: volta a tentar
                    action = make_action(above, gripper_cmd=+1)
                    phase_flip = "GO_ABOVE_OBJ"
                    flip_close_steps = 10
                    flip_shake_steps = 40
                    flip_release_steps = 10

            else:
                raise ValueError(f"policy desconhecida: {policy}")

            obs, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            n += 1

            frame = env.unwrapped.robot.sim.render(width=width, height=height)
            writer.append_data(frame)

    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--policy",
        choices=[
            "random",
            "reach_goal",
            "pick_and_place_phases",
            "push_phases",
            "slide_phases",
            "stack_phases",
            "flip_phases",
        ],
        default="random",
    )
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    record_episode(
        args.env,
        args.out,
        args.policy,
        args.steps,
        args.seed,
        args.fps,
        args.width,
        args.height,
    )
