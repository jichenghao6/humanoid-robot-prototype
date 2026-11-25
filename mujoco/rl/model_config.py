"""Pure MuJoCo/NumPy robot RL configuration shared between GPU training and CPU playback."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT_DIR / "scene.xml"

# 21 core locomotion & arm-swing DOFs (distal wrist/neck/head/upper-torso stay held at neutral 0)
ACTION_JOINT_NAMES: tuple[str, ...] = (
    "spine_lower_flex",
    "spine_lower_abd",
    "spine_lower_twist",
    "hip_L_flex",
    "hip_L_abd",
    "hip_L_twist",
    "knee_L_flex",
    "ankle_L_flex",
    "ankle_L_abd",
    "hip_R_flex",
    "hip_R_abd",
    "hip_R_twist",
    "knee_R_flex",
    "ankle_R_flex",
    "ankle_R_abd",
    "shoulder_L_flex",
    "shoulder_L_abd",
    "elbow_L_flex",
    "shoulder_R_flex",
    "shoulder_R_abd",
    "elbow_R_flex",
)

DEFAULT_ANGLES_RAD: dict[str, float] = {
    "spine_lower_flex": 0.04,
    "hip_L_flex": 0.22,
    "knee_L_flex": 0.44,
    "ankle_L_flex": 0.22,
    "hip_R_flex": 0.22,
    "knee_R_flex": 0.44,
    "ankle_R_flex": 0.22,
    "shoulder_L_abd": 0.08,
    "elbow_L_flex": 0.25,
    "shoulder_R_abd": 0.08,
    "elbow_R_flex": 0.25,
}

# Reward target posture when standing / straight-legged stance (keeps DEFAULT_ANGLES_RAD compatible with v1 checkpoints)
NATURAL_STAND_RAD: dict[str, float] = {
    "spine_lower_flex": 0.03,
    "hip_L_flex": 0.09,
    "knee_L_flex": 0.18,
    "ankle_L_flex": 0.09,
    "hip_R_flex": 0.09,
    "knee_R_flex": 0.18,
    "ankle_R_flex": 0.09,
    "shoulder_L_abd": 0.05,
    "elbow_L_flex": 0.30,
    "shoulder_R_abd": 0.05,
    "elbow_R_flex": 0.30,
}

# RL PD tuning (dt = 0.004, implicitfast, 50 Hz control)
RL_PD_GAINS: dict[str, tuple[float, float]] = {
    "spine_lower_flex": (160.0, 8.0),
    "spine_lower_abd": (150.0, 8.0),
    "spine_lower_twist": (120.0, 6.0),
    "spine_upper_flex": (140.0, 7.0),
    "spine_upper_twist": (120.0, 6.0),
    "neck_flex": (30.0, 2.0),
    "neck_twist": (30.0, 2.0),
    "head_flex": (30.0, 2.0),
    "head_abd": (30.0, 2.0),
    "shoulder_L_flex": (60.0, 3.5),
    "shoulder_L_abd": (60.0, 3.5),
    "shoulder_L_twist": (40.0, 2.5),
    "elbow_L_flex": (50.0, 3.0),
    "wrist_L_flex": (25.0, 1.5),
    "wrist_L_abd": (25.0, 1.5),
    "shoulder_R_flex": (60.0, 3.5),
    "shoulder_R_abd": (60.0, 3.5),
    "shoulder_R_twist": (40.0, 2.5),
    "elbow_R_flex": (50.0, 3.0),
    "wrist_R_flex": (25.0, 1.5),
    "wrist_R_abd": (25.0, 1.5),
    "hip_L_flex": (200.0, 10.0),
    "hip_L_abd": (180.0, 9.0),
    "hip_L_twist": (120.0, 6.0),
    "knee_L_flex": (220.0, 11.0),
    "ankle_L_flex": (90.0, 5.5),
    "ankle_L_abd": (55.0, 4.0),
    "hip_R_flex": (200.0, 10.0),
    "hip_R_abd": (180.0, 9.0),
    "hip_R_twist": (120.0, 6.0),
    "knee_R_flex": (220.0, 11.0),
    "ankle_R_flex": (90.0, 5.5),
    "ankle_R_abd": (55.0, 4.0),
}


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates 3D vector v by inverse of quaternion q [w, x, y, z]."""
    q_w = float(q[0])
    q_vec = np.asarray(q[1:4], dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    a = v64 * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v64) * (2.0 * q_w)
    c = q_vec * (2.0 * float(np.dot(q_vec, v64)))
    return (a - b + c).astype(np.float32)


def configure_mj_model_for_rl(xml_path: Path = SCENE_XML) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
    """Loads scene.xml, configures RL solver/PD parameters, and computes default crouched qpos."""
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.timestep = 0.004
    m.opt.iterations = 8
    m.opt.ls_iterations = 8

    # Lower passive dry friction and excessive joint damping so policy controls cleanly
    m.dof_frictionloss[6:] = 0.0
    m.dof_damping[6:] = np.minimum(m.dof_damping[6:], 0.8)

    default_ctrl = np.zeros(m.nu, dtype=np.float32)
    for aid in range(m.nu):
        jnt_id = int(m.actuator_trnid[aid, 0])
        jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or ""
        kp, kv = RL_PD_GAINS.get(jnt_name, (60.0, 4.0))
        m.actuator_gainprm[aid, 0] = kp
        m.actuator_biasprm[aid, 1] = -kp
        m.actuator_biasprm[aid, 2] = -kv
        default_ctrl[aid] = DEFAULT_ANGLES_RAD.get(jnt_name, 0.0)

    # Compute exact crouched spawn height so soles sit cleanly on floor (z = 0.002 m)
    d_tmp = mujoco.MjData(m)
    mujoco.mj_resetData(m, d_tmp)
    for aid in range(m.nu):
        jnt_id = int(m.actuator_trnid[aid, 0])
        q_adr = int(m.jnt_qposadr[jnt_id])
        d_tmp.qpos[q_adr] = default_ctrl[aid]
    mujoco.mj_forward(m, d_tmp)

    sole_l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_sole_L")
    sole_r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_sole_R")
    half_z = float(m.geom_size[sole_l_id, 2])
    min_sole_z = min(float(d_tmp.geom_xpos[sole_l_id, 2]), float(d_tmp.geom_xpos[sole_r_id, 2])) - half_z
    d_tmp.qpos[2] -= min_sole_z - 0.002
    mujoco.mj_forward(m, d_tmp)

    default_qpos = d_tmp.qpos.astype(np.float32).copy()
    return m, default_qpos, default_ctrl
