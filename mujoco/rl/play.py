"""Play & evaluate trained RSL-RL Walk/Run policy on standard CPU MuJoCo viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import threading
import time

import mujoco
import numpy as np
from tensordict import TensorDict
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rsl_rl.models import MLPModel

from humanoid_env import (
    ACTION_JOINT_NAMES,
    SCENE_XML,
    configure_mj_model_for_rl,
    quat_rotate_inverse,
)


def load_policy(ckpt_path: Path, model: mujoco.MjModel, device: str = "cpu") -> tuple[MLPModel, dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    num_actions = len(ACTION_JOINT_NAMES)
    obs_dim = 74
    critic_dim = 82

    dummy_td = TensorDict(
        {
            "policy": torch.zeros((1, obs_dim), dtype=torch.float32, device=device),
            "critic": torch.zeros((1, critic_dim), dtype=torch.float32, device=device),
        },
        batch_size=[1],
        device=device,
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        dummy_td,
        obs_groups,
        "actor",
        num_actions,
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    ).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor, ckpt


def build_joint_indices(m: mujoco.MjModel, default_ctrl_np: np.ndarray) -> dict:
    name_to_act: dict[str, int] = {}
    for aid in range(m.nu):
        jid = int(m.actuator_trnid[aid, 0])
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname:
            name_to_act[jname] = aid

    act_qpos_idx = []
    act_qvel_idx = []
    act_ctrl_idx = []
    for jname in ACTION_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        act_qpos_idx.append(int(m.jnt_qposadr[jid]))
        act_qvel_idx.append(int(m.jnt_dofadr[jid]))
        act_ctrl_idx.append(name_to_act[jname])

    return {
        "qpos_idx": np.array(act_qpos_idx, dtype=np.int64),
        "qvel_idx": np.array(act_qvel_idx, dtype=np.int64),
        "ctrl_idx": np.array(act_ctrl_idx, dtype=np.int64),
        "default_act_pos": default_ctrl_np[act_ctrl_idx].copy(),
        "default_ctrl": default_ctrl_np.copy(),
        "ctrl_low": m.actuator_ctrlrange[:, 0].copy(),
        "ctrl_high": m.actuator_ctrlrange[:, 1].copy(),
    }


def make_policy_obs(
    data: mujoco.MjData,
    idx: dict,
    cmd: np.ndarray,
    gait_phase: float,
    last_actions: np.ndarray,
) -> TensorDict:
    quat = torch.tensor(data.qpos[3:7], dtype=torch.float32).unsqueeze(0)
    ang_vel = quat_rotate_inverse(quat, torch.tensor(data.qvel[3:6], dtype=torch.float32).unsqueeze(0)).squeeze(0)
    proj_grav = quat_rotate_inverse(quat, torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)).squeeze(0)

    q_act = torch.tensor(data.qpos[idx["qpos_idx"]] - idx["default_act_pos"], dtype=torch.float32)
    dq_act = torch.tensor(data.qvel[idx["qvel_idx"]], dtype=torch.float32)

    cmd_speed = float(np.linalg.norm(cmd))
    moving = 1.0 if cmd_speed > 0.08 else 0.0
    phase_rad = 2.0 * math.pi * gait_phase
    gait_clock = torch.tensor([math.sin(phase_rad) * moving, math.cos(phase_rad) * moving], dtype=torch.float32)

    cmd_t = torch.tensor([cmd[0], cmd[1], cmd[2] * 0.5], dtype=torch.float32)
    act_t = torch.tensor(last_actions, dtype=torch.float32)

    obs_vec = torch.cat(
        [
            ang_vel * 0.25,
            proj_grav,
            cmd_t,
            gait_clock,
            q_act,
            dq_act * 0.05,
            act_t,
        ],
        dim=-1,
    ).unsqueeze(0)

    return TensorDict({"policy": obs_vec}, batch_size=[1])


def run_eval(
    actor: MLPModel,
    m: mujoco.MjModel,
    default_qpos: np.ndarray,
    default_ctrl: np.ndarray,
    vx: float,
    vy: float,
    wz: float,
    duration_s: float,
    viewer_mode: bool,
    push_force_n: float,
    speed_scale: float = 1.0,
    cam_mode: str = "tracking",
    cam_distance: float = 2.4,
    start_paused: bool = False,
) -> dict[str, float]:
    idx = build_joint_indices(m, default_ctrl)
    data = mujoco.MjData(m)
    data.qpos[:] = default_qpos
    data.ctrl[:] = default_ctrl
    mujoco.mj_forward(m, data)

    decimation = 5
    control_dt = m.opt.timestep * decimation
    action_scale = 0.42
    last_actions = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float32)
    gait_phase = 0.0
    cmd = np.array([vx, vy, wz], dtype=np.float32)
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    track_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "humanoid_robot")

    def step_once(
        step_idx: int,
        bump_remaining: int = 0,
        viewer_pert: mujoco.MjvPerturb | None = None,
    ) -> None:
        nonlocal gait_phase, last_actions
        t_cur = step_idx * control_dt
        obs_td = make_policy_obs(data, idx, cmd, gait_phase, last_actions)
        with torch.inference_mode():
            act = actor(obs_td).squeeze(0).cpu().numpy()
        act = np.clip(act, -3.0, 3.0)
        last_actions = act.astype(np.float32)

        ctrl_target = idx["default_ctrl"].copy()
        ctrl_target[idx["ctrl_idx"]] = idx["default_act_pos"] + action_scale * act
        data.ctrl[:] = np.clip(ctrl_target, idx["ctrl_low"], idx["ctrl_high"])

        for _ in range(decimation):
            data.xfrc_applied[:, :] = 0.0
            if push_force_n > 0.0 and 3.0 <= t_cur < 3.2 and torso_id >= 0:
                data.xfrc_applied[torso_id, 0] = push_force_n
            elif bump_remaining > 0 and torso_id >= 0:
                data.xfrc_applied[torso_id, 0] = 120.0
            if viewer_pert is not None and viewer_pert.active:
                mujoco.mjv_applyPerturbForce(m, data, viewer_pert)
            mujoco.mj_step(m, data)

        freq = 0.90 + 0.35 * min(abs(float(cmd[0])), 2.0)
        gait_phase = (gait_phase + freq * control_dt) % 1.0

    if viewer_mode:
        from mujoco import viewer as mj_viewer

        speed_ladder = [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00]
        state_lock = threading.Lock()
        state = {
            "paused": bool(start_paused),
            "step_req": 0,
            "speed": float(max(0.05, speed_scale)),
            "follow_req": False,
            "zoom_req": False,
            "zoom_close": False,
            "reset_req": False,
            "bump_req": 0,
        }

        def on_key(keycode: int) -> None:
            with state_lock:
                if keycode in (32, 80):  # Space / P
                    state["paused"] = not state["paused"]
                elif keycode in (78, 46):  # N / .
                    state["paused"] = True
                    state["step_req"] += 1
                elif keycode in (45, 74, 333):  # - / J / KP_-
                    smaller = [s for s in speed_ladder if s < state["speed"] - 1e-3]
                    state["speed"] = smaller[-1] if smaller else speed_ladder[0]
                elif keycode in (61, 75, 334):  # = / K / KP_+
                    larger = [s for s in speed_ladder if s > state["speed"] + 1e-3]
                    state["speed"] = larger[0] if larger else speed_ladder[-1]
                elif keycode == 70:  # F
                    state["follow_req"] = True
                elif keycode == 67:  # C
                    state["zoom_close"] = not state["zoom_close"]
                    state["zoom_req"] = True
                elif keycode == 82:  # R
                    state["reset_req"] = True
                elif keycode == 66:  # B
                    state["bump_req"] = 10  # 0.20s @ 50Hz

        def apply_follow_cam(cam: mujoco.MjvCamera, dist: float) -> None:
            if track_id >= 0:
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = track_id
                cam.distance = dist
                cam.azimuth = 125.0
                cam.elevation = -12.0

        print(
            "Viewer hotkeys: [Space/P] Pause  [N] Single-Step  [-/= or J/K] Slow/Fast  "
            "[F] Follow Robot  [C] Zoom Close/Full  [B] 120N Bump  [R] Reset"
        )
        with mj_viewer.launch_passive(m, data, key_callback=on_key) as v:
            with v.lock():
                if track_id >= 0:
                    v.cam.trackbodyid = track_id
                if cam_mode == "tracking":
                    apply_follow_cam(v.cam, cam_distance)
                elif cam_mode != "free":
                    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_mode)
                    if cid >= 0:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        v.cam.fixedcamid = cid

            step_idx = 0
            bump_steps = 0
            while v.is_running():
                t0 = time.perf_counter()
                with state_lock:
                    paused = state["paused"]
                    do_step = (not paused) or (state["step_req"] > 0)
                    if state["step_req"] > 0:
                        state["step_req"] -= 1
                    cur_speed = state["speed"]
                    do_follow = state["follow_req"]
                    state["follow_req"] = False
                    do_zoom = state["zoom_req"]
                    state["zoom_req"] = False
                    zoom_dist = 1.45 if state["zoom_close"] else cam_distance
                    do_reset = state["reset_req"]
                    state["reset_req"] = False
                    if state["bump_req"] > 0:
                        bump_steps = state["bump_req"]
                        state["bump_req"] = 0

                with v.lock():
                    if do_reset:
                        data.qpos[:] = default_qpos
                        data.qvel[:] = 0.0
                        data.ctrl[:] = default_ctrl
                        data.xfrc_applied[:, :] = 0.0
                        last_actions[:] = 0.0
                        gait_phase = 0.0
                        step_idx = 0
                        bump_steps = 0
                        mujoco.mj_forward(m, data)
                    elif do_step:
                        step_once(step_idx, bump_remaining=bump_steps, viewer_pert=v.perturb)
                        step_idx += 1
                        if bump_steps > 0:
                            bump_steps -= 1

                    if track_id >= 0 and v.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING and v.cam.trackbodyid <= 0:
                        v.cam.trackbodyid = track_id
                    if do_follow:
                        apply_follow_cam(v.cam, zoom_dist)
                    if do_zoom:
                        v.cam.distance = zoom_dist
                        if v.cam.type != mujoco.mjtCamera.mjCAMERA_TRACKING:
                            apply_follow_cam(v.cam, zoom_dist)

                quat = torch.tensor(data.qpos[3:7], dtype=torch.float32).unsqueeze(0)
                v_body = quat_rotate_inverse(
                    quat, torch.tensor(data.qvel[0:3], dtype=torch.float32).unsqueeze(0)
                ).squeeze(0)
                act_vx = float(v_body[0])
                cam_type = v.cam.type
                if cam_type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    cam_label = "Tracking"
                elif cam_type == mujoco.mjtCamera.mjCAMERA_FIXED:
                    cname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, v.cam.fixedcamid) or "Fixed"
                    cam_label = f"Cam:{cname}"
                else:
                    cam_label = "Free"

                v.set_texts([
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Playback\nCommand\nRobot\nKeys",
                        f"{'PAUSED' if paused else 'PLAYING'} | {cur_speed:.2f}x | {cam_label}\n"
                        f"target vx={cmd[0]:.2f} wz={cmd[2]:.2f} | actual vx={act_vx:.2f} m/s\n"
                        f"x={data.qpos[0]:.2f}m  y={data.qpos[1]:.2f}m  z={data.qpos[2]:.3f}m\n"
                        "[Space]Pause [N]Step [-/=]Speed [F]Follow [C]Zoom [B]Push [R]Reset",
                    )
                ])
                v.sync()

                target_dt = 0.02 if paused else (control_dt / max(cur_speed, 0.05))
                elapsed = time.perf_counter() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
        return {"final_x": float(data.qpos[0]), "final_z": float(data.qpos[2])}

    total_steps = int(round(duration_s / control_dt))
    vel_x_hist = []
    min_z = 999.0
    for s in range(total_steps):
        step_once(s)
        min_z = min(min_z, float(data.qpos[2]))
        if s > 25:
            quat = torch.tensor(data.qpos[3:7], dtype=torch.float32).unsqueeze(0)
            v_body = quat_rotate_inverse(quat, torch.tensor(data.qvel[0:3], dtype=torch.float32).unsqueeze(0)).squeeze(0)
            vel_x_hist.append(float(v_body[0]))

    return {
        "final_x": float(data.qpos[0]),
        "final_y": float(data.qpos[1]),
        "final_z": float(data.qpos[2]),
        "min_z": min_z,
        "mean_vx": float(np.mean(vel_x_hist)) if vel_x_hist else 0.0,
    }


def main() -> int:
    default_ckpt = THIS_DIR / "checkpoints" / "latest_walk.pt"
    if not default_ckpt.exists():
        default_ckpt = THIS_DIR / "logs" / "latest_walk.pt"
    parser = argparse.ArgumentParser(description="Evaluate or view trained humanoid_robot RL policy")
    parser.add_argument("--checkpoint", type=str, default=str(default_ckpt))
    parser.add_argument("--vx", type=float, default=0.9)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--push", type=float, default=0.0, help="Optional forward push (N) at t=3.0..3.2s")
    parser.add_argument("--viewer", action="store_true", help="Open interactive MuJoCo viewer")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Playback time scale in --viewer (default 0.40x slow-mo in viewer)",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="tracking",
        choices=["tracking", "follow_three_quarter", "follow_side", "follow_front", "follow_back", "free"],
        help="Initial camera mode in --viewer (default: tracking robot)",
    )
    parser.add_argument("--cam-distance", type=float, default=2.4, help="Initial tracking distance (meters)")
    parser.add_argument("--paused", action="store_true", help="Start --viewer paused at t=0")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    speed_scale = args.speed if args.speed is not None else (0.40 if args.viewer else 1.0)
    m, default_qpos, default_ctrl = configure_mj_model_for_rl(SCENE_XML)
    actor, _ = load_policy(ckpt_path, m, device="cpu")
    stats = run_eval(
        actor,
        m,
        default_qpos,
        default_ctrl,
        args.vx,
        args.vy,
        args.wz,
        args.duration,
        args.viewer,
        args.push,
        speed_scale=speed_scale,
        cam_mode=args.cam,
        cam_distance=args.cam_distance,
        start_paused=args.paused,
    )
    print(f"Eval ({ckpt_path.name}, cmd=[{args.vx}, {args.vy}, {args.wz}]):", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
