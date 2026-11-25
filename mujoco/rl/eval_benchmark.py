"""Reproducible CPU benchmark and strict CI gate for Walk/Run RL policies."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Callable

import mujoco
import numpy as np
import onnx

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_config import (
    ACTION_JOINT_NAMES,
    GAIT_FREQ_BASE,
    GAIT_FREQ_MAX_VX,
    GAIT_FREQ_SLOPE,
    SCENE_XML,
    configure_mj_model_for_rl,
    quat_rotate_inverse_np,
)
from play import build_joint_indices, load_policy, make_policy_obs


def check_onnx_contract(ckpt_path: Path) -> None:
    """Validate ONNX graph and exact [1, 74] -> [1, 21] I/O tensor dimensions."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ONNX checkpoint not found: {ckpt_path}")
    model = onnx.load(str(ckpt_path))
    onnx.checker.check_model(model)

    inputs = model.graph.input
    outputs = model.graph.output
    if len(inputs) != 1 or len(outputs) != 1:
        raise AssertionError(f"{ckpt_path.name}: expected 1 input & 1 output, got {len(inputs)} & {len(outputs)}")

    in_dims = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in outputs[0].type.tensor_type.shape.dim]
    if in_dims != [1, 74]:
        raise AssertionError(f"{ckpt_path.name}: expected input shape [1, 74], got {in_dims}")
    if out_dims != [1, 21]:
        raise AssertionError(f"{ckpt_path.name}: expected output shape [1, 21], got {out_dims}")


def rollout(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    m: mujoco.MjModel,
    default_qpos: np.ndarray,
    default_ctrl: np.ndarray,
    cmd: tuple[float, float, float],
    duration_s: float = 6.0,
    yaw_deg: float = 0.0,
    push_x_n: float = 0.0,
    push_y_n: float = 0.0,
) -> dict:
    """Roll out policy on CPU MuJoCo and return tracking, fall, flight, and self-collision metrics."""
    idx = build_joint_indices(m, default_ctrl)
    data = mujoco.MjData(m)
    data.qpos[:] = default_qpos
    yaw_rad = math.radians(yaw_deg)
    data.qpos[3] = math.cos(0.5 * yaw_rad)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = math.sin(0.5 * yaw_rad)
    data.ctrl[:] = default_ctrl
    mujoco.mj_forward(m, data)

    decimation = 5
    control_dt = m.opt.timestep * decimation
    n_steps = int(round(duration_s / control_dt))
    steady_start_step = int(round(2.0 / control_dt))
    action_scale = 0.42
    last_actions = np.zeros(len(ACTION_JOINT_NAMES), dtype=np.float32)
    gait_phase = 0.0
    cmd_arr = np.array(cmd, dtype=np.float32)

    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    touch_l_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "foot_L_touch")])
    touch_r_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "foot_R_touch")])

    vel_x_hist: list[float] = []
    vel_y_hist: list[float] = []
    wz_hist: list[float] = []
    min_z = float(data.qpos[2])
    fell_t: float | None = None

    self_cols_full = 0
    self_cols_steady = 0
    self_col_pairs_full: dict[tuple[str, str], int] = {}
    self_col_pairs_steady: dict[tuple[str, str], int] = {}
    flight_hits = 0
    steady_steps = 0

    for step_idx in range(n_steps):
        t_cur = step_idx * control_dt
        obs = make_policy_obs(data, idx, cmd_arr, gait_phase, last_actions)
        act = np.clip(policy_fn(obs), -3.0, 3.0).astype(np.float32)
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(act)):
            raise AssertionError(f"Non-finite observation or action at step {step_idx} (t={t_cur:.2f}s)")
        last_actions = act

        ctrl_target = idx["default_ctrl"].copy()
        ctrl_target[idx["ctrl_idx"]] = idx["default_act_pos"] + action_scale * act
        data.ctrl[:] = np.clip(ctrl_target, idx["ctrl_low"], idx["ctrl_high"])

        for _ in range(decimation):
            data.xfrc_applied[:, :] = 0.0
            if torso_id >= 0 and 3.0 <= t_cur < 3.2:
                if push_x_n != 0.0:
                    data.xfrc_applied[torso_id, 0] = push_x_n
                if push_y_n != 0.0:
                    data.xfrc_applied[torso_id, 1] = push_y_n
            mujoco.mj_step(m, data)

            for ci in range(data.ncon):
                con = data.contact[ci]
                if int(con.geom1) == floor_geom_id or int(con.geom2) == floor_geom_id:
                    continue
                if float(con.dist) < 0.0:
                    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[int(con.geom1)])) or "?"
                    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[int(con.geom2)])) or "?"
                    pair = tuple(sorted((b1, b2)))
                    self_cols_full += 1
                    self_col_pairs_full[pair] = self_col_pairs_full.get(pair, 0) + 1
                    if step_idx >= steady_start_step:
                        self_cols_steady += 1
                        self_col_pairs_steady[pair] = self_col_pairs_steady.get(pair, 0) + 1

        freq = GAIT_FREQ_BASE + GAIT_FREQ_SLOPE * min(abs(float(cmd_arr[0])), GAIT_FREQ_MAX_VX)
        gait_phase = (gait_phase + freq * control_dt) % 1.0

        z_cur = float(data.qpos[2])
        min_z = min(min_z, z_cur)
        quat = np.asarray(data.qpos[3:7], dtype=np.float64)
        proj_g = quat_rotate_inverse_np(quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            fell_t = t_cur
            break
        if z_cur < 0.65 or float(proj_g[2]) > -0.50:
            fell_t = t_cur
            break

        # Match play.py velocity averaging window (t > 0.50 s -> s > 25)
        if step_idx > 25:
            v_body = quat_rotate_inverse_np(quat, np.asarray(data.qvel[0:3], dtype=np.float64))
            vel_x_hist.append(float(v_body[0]))
            vel_y_hist.append(float(v_body[1]))
            wz_hist.append(float(data.qvel[5]))

        if step_idx >= steady_start_step:
            steady_steps += 1
            if float(data.sensordata[touch_l_adr]) <= 5.0 and float(data.sensordata[touch_r_adr]) <= 5.0:
                flight_hits += 1

    return {
        "final_x": float(data.qpos[0]),
        "final_y": float(data.qpos[1]),
        "final_z": float(data.qpos[2]),
        "min_z": min_z,
        "mean_vx": float(np.mean(vel_x_hist)) if vel_x_hist else 0.0,
        "mean_vy": float(np.mean(vel_y_hist)) if vel_y_hist else 0.0,
        "mean_wz": float(np.mean(wz_hist)) if wz_hist else 0.0,
        "fell_t": fell_t,
        "self_cols_full": self_cols_full,
        "self_cols_steady": self_cols_steady,
        "self_col_pairs_full": self_col_pairs_full,
        "self_col_pairs_steady": self_col_pairs_steady,
        "flight_frac": (flight_hits / steady_steps) if steady_steps > 0 else 0.0,
    }


def run_ci_gate(walk_ckpt: Path, run_ckpt: Path) -> None:
    """Strict non-trivial assertions guarding against standing-still, drift, turn, push, or yaw regressions."""
    check_onnx_contract(walk_ckpt)
    check_onnx_contract(run_ckpt)
    m, dq, dc = configure_mj_model_for_rl(SCENE_XML)
    walk_fn = load_policy(walk_ckpt, device="cpu")
    run_fn = load_policy(run_ckpt, device="cpu")

    # --- Walk CI suite ---
    w_stand = rollout(walk_fn, m, dq, dc, (0.0, 0.0, 0.0), duration_s=4.0)
    print("[CI Walk Stand 4s]", w_stand)
    assert w_stand["fell_t"] is None and w_stand["min_z"] > 0.95
    assert abs(w_stand["mean_vx"]) < 0.08
    assert math.hypot(w_stand["final_x"], w_stand["final_y"]) < 0.15

    w_fwd = rollout(walk_fn, m, dq, dc, (0.9, 0.0, 0.0), duration_s=6.0)
    print("[CI Walk Straight 6s]", w_fwd)
    assert w_fwd["fell_t"] is None and w_fwd["min_z"] > 0.90
    assert 0.72 <= w_fwd["mean_vx"] <= 1.08
    assert w_fwd["final_x"] > 4.2 and abs(w_fwd["final_y"]) < 1.5
    assert w_fwd["self_cols_steady"] <= 10, f"Unexpected Walk steady-state self-collisions: {w_fwd['self_col_pairs_steady']}"

    w_yaw0 = rollout(walk_fn, m, dq, dc, (0.9, 0.0, 0.0), duration_s=4.0, yaw_deg=0.0)
    w_yaw180 = rollout(walk_fn, m, dq, dc, (0.9, 0.0, 0.0), duration_s=4.0, yaw_deg=180.0)
    print("[CI Walk Yaw 0/180 4s]", w_yaw0["final_x"], w_yaw180["final_x"], w_yaw0["mean_vx"], w_yaw180["mean_vx"])
    assert abs(w_yaw0["final_x"] + w_yaw180["final_x"]) < 1e-3
    assert abs(w_yaw0["final_y"] + w_yaw180["final_y"]) < 1e-3
    assert abs(w_yaw0["mean_vx"] - w_yaw180["mean_vx"]) < 1e-3

    w_push = rollout(walk_fn, m, dq, dc, (0.9, 0.0, 0.0), duration_s=6.0, push_x_n=150.0)
    print("[CI Walk Push 150N 6s]", w_push)
    assert w_push["fell_t"] is None and w_push["min_z"] > 0.85 and w_push["final_z"] > 0.95
    assert w_push["mean_vx"] > 0.70

    w_turn_r = rollout(walk_fn, m, dq, dc, (0.8, 0.0, -0.5), duration_s=6.0)
    w_turn_l = rollout(walk_fn, m, dq, dc, (0.8, 0.0, +0.5), duration_s=6.0)
    print("[CI Walk Turns 6s] R:", w_turn_r["mean_wz"], "L:", w_turn_l["mean_wz"])
    assert w_turn_r["fell_t"] is None and w_turn_r["min_z"] > 0.90 and w_turn_r["mean_vx"] > 0.55 and w_turn_r["mean_wz"] < -0.10
    assert w_turn_l["fell_t"] is None and w_turn_l["min_z"] > 0.90 and w_turn_l["mean_vx"] > 0.55 and w_turn_l["mean_wz"] > 0.05

    # --- Run CI suite ---
    r_stand = rollout(run_fn, m, dq, dc, (0.0, 0.0, 0.0), duration_s=4.0)
    print("[CI Run Stand 4s]", r_stand)
    assert r_stand["fell_t"] is None and r_stand["min_z"] > 0.95
    assert abs(r_stand["mean_vx"]) < 0.08
    assert math.hypot(r_stand["final_x"], r_stand["final_y"]) < 0.20

    r_fwd = rollout(run_fn, m, dq, dc, (2.0, 0.0, 0.0), duration_s=6.0)
    print("[CI Run Straight 6s]", r_fwd)
    assert r_fwd["fell_t"] is None and r_fwd["min_z"] > 0.88
    assert 1.70 <= r_fwd["mean_vx"] <= 2.30
    assert r_fwd["final_x"] > 9.5 and abs(r_fwd["final_y"]) < 1.8
    assert r_fwd["self_cols_steady"] <= 15, f"Unexpected Run 2.0 steady-state self-collisions: {r_fwd['self_col_pairs_steady']}"

    r_show = rollout(run_fn, m, dq, dc, (2.2, 0.0, 0.0), duration_s=4.0)
    print("[CI Run Showcase 2.2 4s]", r_show)
    assert r_show["fell_t"] is None and r_show["min_z"] > 0.88 and r_show["mean_vx"] >= 1.95
    assert r_show["self_cols_steady"] == 0, f"Unexpected Run 2.2 steady-state self-collisions: {r_show['self_col_pairs_steady']}"

    r_yaw0 = rollout(run_fn, m, dq, dc, (2.0, 0.0, 0.0), duration_s=4.0, yaw_deg=0.0)
    r_yaw180 = rollout(run_fn, m, dq, dc, (2.0, 0.0, 0.0), duration_s=4.0, yaw_deg=180.0)
    print("[CI Run Yaw 0/180 4s]", r_yaw0["final_x"], r_yaw180["final_x"], r_yaw0["mean_vx"], r_yaw180["mean_vx"])
    assert abs(r_yaw0["final_x"] + r_yaw180["final_x"]) < 1e-3
    assert abs(r_yaw0["final_y"] + r_yaw180["final_y"]) < 1e-3
    assert abs(r_yaw0["mean_vx"] - r_yaw180["mean_vx"]) < 1e-3

    r_push = rollout(run_fn, m, dq, dc, (2.0, 0.0, 0.0), duration_s=6.0, push_x_n=150.0)
    print("[CI Run Push 150N 6s]", r_push)
    assert r_push["fell_t"] is None and r_push["min_z"] > 0.85 and r_push["final_z"] > 0.90
    assert r_push["mean_vx"] > 1.70

    r_turn_l = rollout(run_fn, m, dq, dc, (1.8, 0.0, +0.5), duration_s=6.0)
    r_turn_r = rollout(run_fn, m, dq, dc, (1.8, 0.0, -0.5), duration_s=6.0)
    print("[CI Run Turns 6s] L:", r_turn_l["mean_wz"], "R:", r_turn_r["mean_wz"])
    assert r_turn_l["fell_t"] is None and r_turn_l["min_z"] > 0.88 and r_turn_l["mean_vx"] > 1.45 and r_turn_l["mean_wz"] > 0.25
    assert r_turn_r["fell_t"] is None and r_turn_r["min_z"] > 0.88 and r_turn_r["mean_vx"] > 1.45 and r_turn_r["mean_wz"] < -0.20

    print("ALL_CI_GATES_PASSED")


def run_full_report(walk_ckpt: Path, run_ckpt: Path) -> None:
    """Print full 10s quantitative metrics including full-window vs steady-state (t >= 2s) self-collisions."""
    check_onnx_contract(walk_ckpt)
    check_onnx_contract(run_ckpt)
    m, dq, dc = configure_mj_model_for_rl(SCENE_XML)
    walk_fn = load_policy(walk_ckpt, device="cpu")
    run_fn = load_policy(run_ckpt, device="cpu")

    walk_cases = [
        ("Stand [0,0,0] 8s", (0.0, 0.0, 0.0), 8.0, 0.0),
        ("Slow Walk [0.55,0,0] 10s", (0.55, 0.0, 0.0), 10.0, 0.0),
        ("Normal Walk [0.9,0,0] 10s", (0.9, 0.0, 0.0), 10.0, 0.0),
        ("Fast Walk [1.25,0,0] 10s", (1.25, 0.0, 0.0), 10.0, 0.0),
        ("Walk + 150N Push 10s", (0.9, 0.0, 0.0), 10.0, 150.0),
        ("Walk Turn Left [0.8,0,+0.5] 10s", (0.8, 0.0, +0.5), 10.0, 0.0),
        ("Walk Turn Right [0.8,0,-0.5] 10s", (0.8, 0.0, -0.5), 10.0, 0.0),
        ("Walk Strafe [0,0.2,0] 10s", (0.0, 0.2, 0.0), 10.0, 0.0),
    ]
    print("=== WALK FULL BENCHMARK ===")
    for name, cmd, dur, px in walk_cases:
        res = rollout(walk_fn, m, dq, dc, cmd, duration_s=dur, push_x_n=px)
        print(f"{name}: {res}")

    run_cases = [
        ("Stand [0,0,0] 8s", (0.0, 0.0, 0.0), 8.0, 0.0),
        ("Jog [1.4,0,0] 10s", (1.4, 0.0, 0.0), 10.0, 0.0),
        ("Run [2.0,0,0] 10s", (2.0, 0.0, 0.0), 10.0, 0.0),
        ("Showcase Run [2.2,0,0] 10s", (2.2, 0.0, 0.0), 10.0, 0.0),
        ("Fast Run [2.4,0,0] 10s", (2.4, 0.0, 0.0), 10.0, 0.0),
        ("Run + 150N Push 10s", (2.0, 0.0, 0.0), 10.0, 150.0),
        ("Run Turn Left [1.8,0,+0.5] 10s", (1.8, 0.0, +0.5), 10.0, 0.0),
        ("Run Turn Right [1.8,0,-0.5] 10s", (1.8, 0.0, -0.5), 10.0, 0.0),
        ("Run Strafe [1.6,0.2,0] 10s", (1.6, 0.2, 0.0), 10.0, 0.0),
    ]
    print("=== RUN FULL BENCHMARK ===")
    for name, cmd, dur, px in run_cases:
        res = rollout(run_fn, m, dq, dc, cmd, duration_s=dur, push_x_n=px)
        print(f"{name}: {res}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU MuJoCo benchmark & strict CI gate for Walk/Run ONNX policies")
    parser.add_argument("--ci", action="store_true", help="Run strict CI gate assertions")
    parser.add_argument("--full", action="store_true", help="Run full 10s quantitative report")
    parser.add_argument("--walk-ckpt", type=Path, default=THIS_DIR / "checkpoints" / "latest_walk.onnx")
    parser.add_argument("--run-ckpt", type=Path, default=THIS_DIR / "checkpoints" / "latest_run.onnx")
    args = parser.parse_args(argv)

    if not args.ci and not args.full:
        args.ci = True
    if args.ci:
        run_ci_gate(args.walk_ckpt, args.run_ckpt)
    if args.full:
        run_full_report(args.walk_ckpt, args.run_ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
