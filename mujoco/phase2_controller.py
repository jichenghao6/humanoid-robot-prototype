# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=4.0.1", "numpy"]
# ///
"""Phase 2 Closed-Loop Dynamics Controller for Humanoid Robot in MuJoCo.

Modes:
  - balance       : Bent-knee athletic stance with IMU + CoM closed-loop push recovery
  - squat         : Continuous deep squat & stand cycles under real gravity
  - weight_shift  : Lateral CoM transfer onto left foot + single-leg right foot lift
  - sweep         : Extreme range-of-motion self-collision audit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
SCENE_XML = ROOT_DIR / "scene.xml"
DEG2RAD = math.pi / 180.0


def quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    """Converts MuJoCo quaternion [w, x, y, z] to (roll, pitch, yaw) in radians."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


@dataclass
class ControllerIndices:
    act: dict[str, int]
    jnt_q: dict[str, int]
    jnt_v: dict[str, int]
    floor_id: int
    torso_id: int
    foot_l_id: int
    foot_r_id: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "ControllerIndices":
        act = {}
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                act[name] = i
        jnt_q = {}
        jnt_v = {}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                jnt_q[name] = int(model.jnt_qposadr[i])
                jnt_v[name] = int(model.jnt_dofadr[i])
        return cls(
            act=act,
            jnt_q=jnt_q,
            jnt_v=jnt_v,
            floor_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"),
            torso_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso"),
            foot_l_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L"),
            foot_r_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R"),
        )


class HumanoidPhase2Controller:
    """Closed-loop IMU + CoM + joint-space controller for standing, squats, and single-leg balance."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.idx = ControllerIndices.from_model(model)
        self._apply_controller_gains()

    def _apply_controller_gains(self) -> None:
        """Sets balanced PD servo gains suited for dynamic squats and compliant push recovery."""
        tuning: dict[str, tuple[float, float]] = {
            "spine_lower_flex_servo": (320.0, 28.0),
            "spine_lower_abd_servo": (300.0, 26.0),
            "spine_lower_twist_servo": (220.0, 20.0),
            "spine_upper_flex_servo": (280.0, 24.0),
            "spine_upper_twist_servo": (200.0, 18.0),
            "hip_L_flex_servo": (460.0, 38.0),
            "hip_R_flex_servo": (460.0, 38.0),
            "hip_L_abd_servo": (420.0, 34.0),
            "hip_R_abd_servo": (420.0, 34.0),
            "hip_L_twist_servo": (240.0, 20.0),
            "hip_R_twist_servo": (240.0, 20.0),
            "knee_L_flex_servo": (480.0, 40.0),
            "knee_R_flex_servo": (480.0, 40.0),
            "ankle_L_flex_servo": (420.0, 36.0),
            "ankle_R_flex_servo": (420.0, 36.0),
            "ankle_L_abd_servo": (280.0, 24.0),
            "ankle_R_abd_servo": (280.0, 24.0),
        }
        for name, (kp, kv) in tuning.items():
            if name in self.idx.act:
                aid = self.idx.act[name]
                self.model.actuator_gainprm[aid, 0] = kp
                self.model.actuator_biasprm[aid, 1] = -kp
                self.model.actuator_biasprm[aid, 2] = -kv

    def set_servo_deg(self, data: mujoco.MjData, servo_name: str, deg: float) -> None:
        aid = self.idx.act[servo_name]
        lo, hi = self.model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(deg * DEG2RAD, lo, hi))

    def step_balance(
        self,
        data: mujoco.MjData,
        t: float,
        mode: str = "balance",
    ) -> dict[str, float]:
        """Computes one closed-loop control update using IMU, CoM, and foot touch feedback."""
        data.ctrl[:] = 0.0

        # Base athletic crouch angles (deg)
        crouch_deg = 14.0
        com_y_offset = 0.0
        com_x_offset = 0.032  # Center CoM slightly ahead of ankle (-0.020) over midfoot (+0.012..+0.035)
        right_leg_lift = 0.0
        torso_roll_bias = 0.0
        pelvis_roll_bias = 0.0
        weight_shift_u = 0.0

        if mode == "squat":
            # Smooth 3.2s squat cycle: 12 deg -> 38 deg hip crouch (knee 24 -> 76 deg)
            s = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / 3.2))
            crouch_deg = 12.0 + 26.0 * s
            com_x_offset = 0.028 + 0.015 * s
            # Slight counter arm reach for natural counterbalance
            self.set_servo_deg(data, "shoulder_L_flex_servo", 24.0 * s)
            self.set_servo_deg(data, "shoulder_R_flex_servo", 24.0 * s)
            self.set_servo_deg(data, "elbow_L_flex_servo", 12.0 + 28.0 * s)
            self.set_servo_deg(data, "elbow_R_flex_servo", 12.0 + 28.0 * s)
        elif mode == "weight_shift":
            # 7s single-leg balance cycle: shift onto left foot -> lift right foot -> set down
            phase = t % 7.0
            if phase < 2.0:
                weight_shift_u = 0.5 * (1.0 - math.cos(math.pi * phase / 2.0))
            elif phase < 5.0:
                weight_shift_u = 1.0
            else:
                weight_shift_u = 0.5 * (1.0 + math.cos(math.pi * (phase - 5.0) / 2.0))

            if 1.8 < phase < 5.2:
                lift_u = math.sin(math.pi * (phase - 1.8) / 3.4)
                right_leg_lift = max(0.0, lift_u) ** 0.85

            crouch_deg = 13.0
            # Negative rotation around +X (front) leans torso & pelvis toward +Y (left)!
            torso_roll_bias = -14.0 * weight_shift_u
            pelvis_roll_bias = -6.5 * weight_shift_u
            self.set_servo_deg(data, "shoulder_L_abd_servo", 24.0 * weight_shift_u)
            self.set_servo_deg(data, "shoulder_R_abd_servo", 2.0)
        else:
            # Slight relaxed arms in balance mode
            self.set_servo_deg(data, "shoulder_L_abd_servo", 4.0)
            self.set_servo_deg(data, "shoulder_R_abd_servo", 4.0)
            self.set_servo_deg(data, "elbow_L_flex_servo", 10.0)
            self.set_servo_deg(data, "elbow_R_flex_servo", 10.0)

        # Read IMU + CoM + foot sole references
        imu_quat = data.sensor("pelvis_quat").data
        gyro = data.sensor("pelvis_gyro").data  # [wx, wy, wz] in IMU frame
        com = data.sensor("robot_com").data

        roll, pitch, _ = quat_to_rpy(imu_quat)
        roll_rate = float(gyro[0])
        pitch_rate = float(gyro[1])

        sole_l = data.site("foot_L_sole_site").xpos
        sole_r = data.site("foot_R_sole_site").xpos
        ankle_l = data.xpos[self.idx.foot_l_id]
        ankle_r = data.xpos[self.idx.foot_r_id]

        if mode == "weight_shift":
            # Interpolate support target from midpoint (0) to left sole center (ankle_l)
            u = weight_shift_u
            support_x = float(ankle_l[0]) * u + 0.5 * (float(ankle_l[0]) + float(ankle_r[0])) * (1.0 - u)
            support_y = float(ankle_l[1]) * u + 0.5 * (float(ankle_l[1]) + float(ankle_r[1])) * (1.0 - u)
            com_y_offset = 0.004 * u  # Bias slightly toward outer foot edge for robust margin
            com_y_feedforward = float(ankle_l[1]) * u
        else:
            support_x = 0.5 * (float(ankle_l[0]) + float(ankle_r[0]))
            support_y = 0.5 * (float(ankle_l[1]) + float(ankle_r[1]))
            com_y_feedforward = 0.0

        err_com_x = (float(com[0]) - support_x) - com_x_offset
        err_com_y = (float(com[1]) - support_y) - com_y_offset

        # Closed-loop pitch stabilization (ankle + hip + waist coordination)
        # Positive pitch = tilting forward (+X); positive ankle_flex bends shin forward (+X).
        # To resist forward tilt (+pitch > 0, +err_com_x > 0), drive ankle_flex negative (plantarflex push-back)!
        pitch_fb_deg = (
            -55.0 * pitch
            - 11.0 * pitch_rate
            - 120.0 * err_com_x
        )
        hip_pitch_fb_deg = (
            -28.0 * pitch
            - 5.0 * pitch_rate
            - 45.0 * err_com_x
        )

        # Closed-loop roll stabilization
        # Positive roll = leaning right (-Y); to push CoM left (+Y), increase err_com_y gain.
        # In MuJoCo: hip_L_abd (+X axis) moves left leg +Y (leans pelvis -Y); hip_R_abd (-X axis) moves right leg -Y (leans pelvis +Y).
        roll_target = pelvis_roll_bias * DEG2RAD
        roll_fb_deg = (
            -45.0 * (roll - roll_target)
            - 8.0 * roll_rate
            + 135.0 * err_com_y
        )

        # Left leg (always support leg in weight_shift)
        hip_l_flex = crouch_deg + hip_pitch_fb_deg
        knee_l_flex = 2.0 * crouch_deg
        ankle_l_flex = crouch_deg + pitch_fb_deg - 2.0

        # Right leg (lifts during weight_shift)
        hip_r_flex = crouch_deg + hip_pitch_fb_deg + 34.0 * right_leg_lift
        knee_r_flex = 2.0 * crouch_deg + 54.0 * right_leg_lift
        ankle_r_flex = (
            crouch_deg + pitch_fb_deg - 2.0
        ) * (1.0 - right_leg_lift) + (20.0) * right_leg_lift

        # Lateral hip/ankle abduction for CoM shifting + roll feedback
        # To shift pelvis toward +Y (com_y_feedforward > 0): hip_L_abd < 0 (adduct), ankle_L_abd > 0
        lat_shift_deg = (com_y_feedforward / 0.19) * 8.0
        hip_l_abd = -lat_shift_deg + roll_fb_deg
        hip_r_abd = lat_shift_deg - roll_fb_deg - 2.0 * right_leg_lift
        ankle_l_abd = lat_shift_deg - 0.85 * roll_fb_deg
        ankle_r_abd = (-lat_shift_deg + 0.65 * roll_fb_deg) * (1.0 - right_leg_lift)

        self.set_servo_deg(data, "hip_L_flex_servo", hip_l_flex)
        self.set_servo_deg(data, "knee_L_flex_servo", knee_l_flex)
        self.set_servo_deg(data, "ankle_L_flex_servo", ankle_l_flex)
        self.set_servo_deg(data, "hip_L_abd_servo", hip_l_abd)
        self.set_servo_deg(data, "ankle_L_abd_servo", ankle_l_abd)

        self.set_servo_deg(data, "hip_R_flex_servo", hip_r_flex)
        self.set_servo_deg(data, "knee_R_flex_servo", knee_r_flex)
        self.set_servo_deg(data, "ankle_R_flex_servo", ankle_r_flex)
        self.set_servo_deg(data, "hip_R_abd_servo", hip_r_abd)
        self.set_servo_deg(data, "ankle_R_abd_servo", ankle_r_abd)

        # Keep torso upright and softly compensate CoM lean
        self.set_servo_deg(data, "spine_lower_flex_servo", 4.0 - 20.0 * pitch - 3.0 * pitch_rate)
        self.set_servo_deg(data, "spine_upper_flex_servo", 2.0 - 10.0 * pitch)
        self.set_servo_deg(data, "spine_lower_abd_servo", torso_roll_bias - 22.0 * (roll - roll_target))
        self.set_servo_deg(data, "neck_flex_servo", -4.0)

        return {
            "pelvis_z": float(data.qpos[2]),
            "pitch_deg": pitch / DEG2RAD,
            "roll_deg": roll / DEG2RAD,
            "com_x": float(com[0]),
            "com_y": float(com[1]),
            "touch_L": float(data.sensor("foot_L_touch").data[0]),
            "touch_R": float(data.sensor("foot_R_touch").data[0]),
            "sole_L_z": float(sole_l[2]),
            "sole_R_z": float(sole_r[2]),
        }


def run_headless_suite() -> int:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    ctrl = HumanoidPhase2Controller(model)

    # 1. Long 10s closed-loop balance + external impulse push recovery test
    data = mujoco.MjData(model)
    for step in range(5000):  # 10s
        t = step * model.opt.timestep
        # Apply 120 N forward push on torso at t=2.0..2.2s, and 90 N lateral push at t=5.5..5.7s
        data.xfrc_applied[ctrl.idx.torso_id, :] = 0.0
        if 2.0 <= t < 2.2:
            data.xfrc_applied[ctrl.idx.torso_id, 0] = 120.0
        elif 5.5 <= t < 5.7:
            data.xfrc_applied[ctrl.idx.torso_id, 1] = 90.0

        info = ctrl.step_balance(data, t, mode="balance")
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or info["pelvis_z"] < 0.70:
            raise RuntimeError(f"Balance failed at t={t:.2f}s: {info}")

    print(
        f"[Phase 2 - 1] 10s Closed-loop balance + 120N/90N pushes: PASS | "
        f"pelvis_z={info['pelvis_z']:.3f}m, pitch={info['pitch_deg']:+.2f}deg, "
        f"roll={info['roll_deg']:+.2f}deg, touch=({info['touch_L']:.0f}N, {info['touch_R']:.0f}N)"
    )

    # 2. Squat test (2 full cycles = 6.4s)
    data = mujoco.MjData(model)
    min_z = 999.0
    max_z = -999.0
    min_touch = 9999.0
    for step in range(3200):
        t = step * model.opt.timestep
        info = ctrl.step_balance(data, t, mode="squat")
        mujoco.mj_step(model, data)
        min_z = min(min_z, info["pelvis_z"])
        max_z = max(max_z, info["pelvis_z"])
        if t > 0.5:
            min_touch = min(min_touch, info["touch_L"], info["touch_R"])
        if not np.all(np.isfinite(data.qpos)) or info["pelvis_z"] < 0.55:
            raise RuntimeError(f"Squat failed at t={t:.2f}s: {info}")

    print(
        f"[Phase 2 - 2] 6.4s Dynamic squat cycles: PASS | "
        f"pelvis_z range=[{min_z:.3f}m, {max_z:.3f}m], min_foot_touch={min_touch:.0f}N"
    )

    # 3. Single-leg weight shift test (7s)
    data = mujoco.MjData(model)
    max_right_lift = 0.0
    min_right_touch = 9999.0
    for step in range(3500):
        t = step * model.opt.timestep
        info = ctrl.step_balance(data, t, mode="weight_shift")
        mujoco.mj_step(model, data)
        if 1.8 <= t <= 5.0:
            max_right_lift = max(max_right_lift, info["sole_R_z"])
            min_right_touch = min(min_right_touch, info["touch_R"])
        if not np.all(np.isfinite(data.qpos)) or info["pelvis_z"] < 0.70:
            raise RuntimeError(f"Weight-shift failed at t={t:.2f}s: {info}")

    print(
        f"[Phase 2 - 3] 7.0s Single-leg weight shift: PASS | "
        f"right_sole_max_z={max_right_lift:.3f}m, right_touch_min={min_right_touch:.1f}N, "
        f"final_pelvis_z={info['pelvis_z']:.3f}m"
    )

    # 4. Self-collision ROM sweep (pin floating base only so actuated joints reach full ROM)
    data = mujoco.MjData(model)
    data.qpos[2] = 1.5
    unexpected_self_hits: set[tuple[str, str]] = set()
    max_hip_l = 0.0
    max_knee_l = 0.0
    max_sh_l = 0.0
    max_sh_abd_l = 0.0
    for step in range(400):
        s = step / 400.0
        data.qpos[0:3] = [0.0, 0.0, 1.5]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:6] = 0.0
        ctrl.set_servo_deg(data, "hip_L_flex_servo", 85.0 * math.sin(math.pi * s))
        ctrl.set_servo_deg(data, "knee_L_flex_servo", 110.0 * s)
        ctrl.set_servo_deg(data, "hip_R_flex_servo", -25.0 * math.sin(math.pi * s))
        ctrl.set_servo_deg(data, "knee_R_flex_servo", 90.0 * s)
        ctrl.set_servo_deg(data, "spine_lower_flex_servo", 20.0 * math.sin(2.0 * math.pi * s))
        ctrl.set_servo_deg(data, "spine_lower_twist_servo", 22.0 * math.cos(2.0 * math.pi * s))
        ctrl.set_servo_deg(data, "shoulder_L_flex_servo", 120.0 * s)
        ctrl.set_servo_deg(data, "shoulder_L_abd_servo", 80.0 * s)
        ctrl.set_servo_deg(data, "shoulder_R_flex_servo", 120.0 * s)
        ctrl.set_servo_deg(data, "shoulder_R_abd_servo", 80.0 * s)
        mujoco.mj_step(model, data)
        max_hip_l = max(max_hip_l, abs(float(data.qpos[ctrl.idx.jnt_q["hip_L_flex"]]) / DEG2RAD))
        max_knee_l = max(max_knee_l, abs(float(data.qpos[ctrl.idx.jnt_q["knee_L_flex"]]) / DEG2RAD))
        max_sh_l = max(max_sh_l, abs(float(data.qpos[ctrl.idx.jnt_q["shoulder_L_flex"]]) / DEG2RAD))
        max_sh_abd_l = max(max_sh_abd_l, abs(float(data.qpos[ctrl.idx.jnt_q["shoulder_L_abd"]]) / DEG2RAD))
        for ci in range(data.ncon):
            c = data.contact[ci]
            if c.geom1 != ctrl.idx.floor_id and c.geom2 != ctrl.idx.floor_id:
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
                pair = tuple(sorted((g1, g2)))
                unexpected_self_hits.add(pair)

    print(
        f"[Phase 2 - 4] ROM self-collision sweep: "
        f"peak_deg(hip_L={max_hip_l:.1f}, knee_L={max_knee_l:.1f}, sh_L={max_sh_l:.1f}, sh_abd_L={max_sh_abd_l:.1f}), "
        f"hits={sorted(unexpected_self_hits)}"
    )
    if max_hip_l < 70.0 or max_knee_l < 80.0 or max_sh_l < 80.0:
        raise RuntimeError(
            f"ROM sweep did not reach expected angles: hip_L={max_hip_l:.1f}, knee_L={max_knee_l:.1f}, sh_L={max_sh_l:.1f}"
        )
    if unexpected_self_hits:
        raise RuntimeError(f"Unexpected self-collisions during ROM sweep: {sorted(unexpected_self_hits)}")
    return 0


def run_viewer(mode: str) -> int:
    from mujoco import viewer as mj_viewer

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    ctrl = HumanoidPhase2Controller(model)
    data = mujoco.MjData(model)

    with mj_viewer.launch_passive(model, data) as v:
        t_sim = 0.0
        while v.is_running():
            step_start = time.perf_counter()
            for _ in range(8):  # 8 * 0.002s = 16ms (~60 Hz render)
                ctrl.step_balance(data, t_sim, mode=mode)
                mujoco.mj_step(model, data)
                t_sim += model.opt.timestep
            v.sync()
            elapsed = time.perf_counter() - step_start
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 closed-loop MuJoCo controller")
    parser.add_argument(
        "--mode",
        choices=["balance", "squat", "weight_shift"],
        default="balance",
        help="Active closed-loop mode when launching viewer",
    )
    parser.add_argument("--viewer", action="store_true", help="Launch live interactive MuJoCo viewer")
    args = parser.parse_args()

    if args.viewer:
        return run_viewer(args.mode)
    return run_headless_suite()


if __name__ == "__main__":
    sys.exit(main())
