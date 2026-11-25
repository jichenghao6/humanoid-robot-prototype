"""MuJoCo-Warp GPU-accelerated VecEnv for bipedal walking & running (RSL-RL 6.0)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

import mujoco
import mujoco_warp as mjw
import numpy as np
from rsl_rl.env import VecEnv
from tensordict import TensorDict
import torch
import warp as wp

try:
    from .model_config import (
        ACTION_JOINT_NAMES,
        DEFAULT_ANGLES_RAD,
        NATURAL_STAND_RAD,
        RL_PD_GAINS,
        ROOT_DIR,
        SCENE_XML,
        configure_mj_model_for_rl,
    )
except ImportError:
    from model_config import (
        ACTION_JOINT_NAMES,
        DEFAULT_ANGLES_RAD,
        NATURAL_STAND_RAD,
        RL_PD_GAINS,
        ROOT_DIR,
        SCENE_XML,
        configure_mj_model_for_rl,
    )


@dataclass
class EnvConfig:
    task: Literal["walk", "run"] = "walk"
    num_envs: int = 1024
    episode_length_s: float = 20.0
    sim_dt: float = 0.004
    decimation: int = 5
    action_scale: float = 0.42
    nconmax_per_env: int = 16
    njmax_per_env: int = 64
    enable_pushes: bool = True
    push_interval_s: float = 6.0
    obs_noise: bool = True


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotates vector v (N, 3) by the inverse of quaternion q (N, 4) [w, x, y, z]."""
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * (2.0 * q_w)
    c = q_vec * (2.0 * torch.sum(q_vec * v, dim=-1, keepdim=True))
    return a - b + c


class HumanoidLocomotionEnv(VecEnv):
    """GPU-vectorized MuJoCo-Warp locomotion environment compatible with rsl_rl 6.0."""

    def __init__(self, env_cfg: EnvConfig, device: str = "cuda:0") -> None:
        self.cfg_obj = env_cfg
        self.cfg = env_cfg.__dict__
        self.device = torch.device(device)
        self.num_envs = env_cfg.num_envs
        self.num_actions = len(ACTION_JOINT_NAMES)
        self.control_dt = env_cfg.sim_dt * env_cfg.decimation
        self.max_episode_length = int(round(env_cfg.episode_length_s / self.control_dt))

        self.mj_model, default_qpos_np, default_ctrl_np = configure_mj_model_for_rl(SCENE_XML)
        self.mj_model.opt.timestep = env_cfg.sim_dt

        wp.init()
        self.mw_model = mjw.put_model(self.mj_model)
        if hasattr(self.mw_model.opt, "warn_overflow"):
            self.mw_model.opt.warn_overflow = 0
        self.mw_data = mjw.make_data(
            self.mj_model,
            nworld=self.num_envs,
            nconmax=env_cfg.nconmax_per_env,
            njmax=env_cfg.njmax_per_env,
        )

        # Zero-copy GPU views into MuJoCo-Warp buffers
        self.qpos = wp.to_torch(self.mw_data.qpos)
        self.qvel = wp.to_torch(self.mw_data.qvel)
        self.ctrl = wp.to_torch(self.mw_data.ctrl)
        self.xpos = wp.to_torch(self.mw_data.xpos)
        self.xquat = wp.to_torch(self.mw_data.xquat)
        self.sensordata = wp.to_torch(self.mw_data.sensordata)
        self.site_xpos = wp.to_torch(self.mw_data.site_xpos)

        # Resolve joint/actuator/body/sensor indices
        self._build_index_mappings(default_qpos_np, default_ctrl_np)

        # Allocate RL state buffers on CUDA
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.gait_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.feet_air_time = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.last_contacts = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.foot_prev_xy = torch.zeros((self.num_envs, 2, 2), dtype=torch.float32, device=self.device)

        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)

        # Reset all environments initially
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_ids)
        mjw.forward(self.mw_model, self.mw_data)
        wp.synchronize()
        self.foot_prev_xy[:, 0, :] = self.site_xpos[:, self.sole_l_site_id, :2]
        self.foot_prev_xy[:, 1, :] = self.site_xpos[:, self.sole_r_site_id, :2]

        self._obs_td = self._compute_observations()

    def _build_index_mappings(self, default_qpos_np: np.ndarray, default_ctrl_np: np.ndarray) -> None:
        m = self.mj_model
        self.default_qpos = torch.tensor(default_qpos_np, dtype=torch.float32, device=self.device)
        self.default_ctrl = torch.tensor(default_ctrl_np, dtype=torch.float32, device=self.device)

        act_qpos_idx = []
        act_qvel_idx = []
        act_ctrl_idx = []
        jnt_low = []
        jnt_high = []

        name_to_act: dict[str, int] = {}
        for aid in range(m.nu):
            jid = int(m.actuator_trnid[aid, 0])
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname:
                name_to_act[jname] = aid

        for jname in ACTION_JOINT_NAMES:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = name_to_act[jname]
            act_qpos_idx.append(int(m.jnt_qposadr[jid]))
            act_qvel_idx.append(int(m.jnt_dofadr[jid]))
            act_ctrl_idx.append(aid)
            jnt_low.append(float(m.jnt_range[jid, 0]))
            jnt_high.append(float(m.jnt_range[jid, 1]))

        self.act_qpos_idx = torch.tensor(act_qpos_idx, dtype=torch.long, device=self.device)
        self.act_qvel_idx = torch.tensor(act_qvel_idx, dtype=torch.long, device=self.device)
        self.act_ctrl_idx = torch.tensor(act_ctrl_idx, dtype=torch.long, device=self.device)
        self.default_act_pos = self.default_qpos[self.act_qpos_idx]
        self.natural_stand_pos = torch.tensor(
            [NATURAL_STAND_RAD.get(jname, 0.0) for jname in ACTION_JOINT_NAMES],
            dtype=torch.float32,
            device=self.device,
        )

        self.jnt_low = torch.tensor(jnt_low, dtype=torch.float32, device=self.device)
        self.jnt_high = torch.tensor(jnt_high, dtype=torch.float32, device=self.device)

        ctrl_low = m.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_high = m.actuator_ctrlrange[:, 1].astype(np.float32)
        self.ctrl_low = torch.tensor(ctrl_low, dtype=torch.float32, device=self.device)
        self.ctrl_high = torch.tensor(ctrl_high, dtype=torch.float32, device=self.device)

        # PD parameters for torque penalty estimation
        self.kp_act = torch.tensor(m.actuator_gainprm[act_ctrl_idx, 0], dtype=torch.float32, device=self.device)
        self.kv_act = torch.tensor(-m.actuator_biasprm[act_ctrl_idx, 2], dtype=torch.float32, device=self.device)
        self.tau_limit_act = torch.tensor(m.actuator_forcerange[act_ctrl_idx, 1], dtype=torch.float32, device=self.device)

        # Body & site indices
        self.torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.foot_l_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
        self.foot_r_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_R")
        self.sole_l_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_L_sole_site")
        self.sole_r_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_R_sole_site")

        # Sensor addresses (order from model.xml: pelvis_quat[4], gyro[3], acc[3], com[3], touch_L[1], touch_R[1])
        self.touch_l_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "foot_L_touch")])
        self.touch_r_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "foot_R_touch")])

        # Non-foot body IDs for fall/ground-hit termination checks
        non_foot_bodies = [
            "torso",
            "lumbar_waist",
            "head",
            "thigh_L",
            "thigh_R",
            "shin_L",
            "shin_R",
            "upper_arm_L",
            "upper_arm_R",
            "forearm_L",
            "forearm_R",
            "hand_L",
            "hand_R",
        ]
        self.non_foot_body_ids = torch.tensor(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in non_foot_bodies],
            dtype=torch.long,
            device=self.device,
        )

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        n = env_ids.shape[0]
        if n == 0:
            return
        if self.cfg_obj.task == "walk":
            r = torch.rand(n, device=self.device)
            vx = torch.empty(n, device=self.device).uniform_(0.45, 1.35)
            vx = torch.where(r < 0.10, torch.zeros_like(vx), vx)
            vx = torch.where((r >= 0.10) & (r < 0.18), torch.empty(n, device=self.device).uniform_(-0.3, 0.3), vx)
            vy = torch.empty(n, device=self.device).uniform_(-0.20, 0.20)
            wz = torch.empty(n, device=self.device).uniform_(-0.6, 0.6)
        else:
            vx = torch.empty(n, device=self.device).uniform_(1.2, 2.6)
            r = torch.rand(n, device=self.device)
            vx = torch.where(r < 0.06, torch.zeros_like(vx), vx)
            vy = torch.empty(n, device=self.device).uniform_(-0.22, 0.22)
            wz = torch.empty(n, device=self.device).uniform_(-0.6, 0.6)

        stand_mask = (torch.abs(vx) < 0.12) & (torch.abs(vy) < 0.08) & (torch.abs(wz) < 0.12)
        vx = torch.where(stand_mask, torch.zeros_like(vx), vx)
        vy = torch.where(stand_mask, torch.zeros_like(vy), vy)
        wz = torch.where(stand_mask, torch.zeros_like(wz), wz)

        self.commands[env_ids, 0] = vx
        self.commands[env_ids, 1] = vy
        self.commands[env_ids, 2] = wz

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        n = env_ids.shape[0]
        if n == 0:
            return

        qpos_reset = self.default_qpos.unsqueeze(0).repeat(n, 1)
        qpos_reset[:, self.act_qpos_idx] += torch.empty((n, self.num_actions), device=self.device).uniform_(-0.04, 0.04)
        # Small random initial yaw
        yaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        qpos_reset[:, 3] = torch.cos(0.5 * yaw)
        qpos_reset[:, 4] = 0.0
        qpos_reset[:, 5] = 0.0
        qpos_reset[:, 6] = torch.sin(0.5 * yaw)

        qvel_reset = torch.zeros((n, self.mj_model.nv), dtype=torch.float32, device=self.device)
        qvel_reset[:, 0:3] = torch.empty((n, 3), device=self.device).uniform_(-0.05, 0.05)

        self.qpos[env_ids] = qpos_reset
        self.qvel[env_ids] = qvel_reset
        self.ctrl[env_ids] = self.default_ctrl.unsqueeze(0).expand(n, -1)

        self.episode_length_buf[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.gait_phase[env_ids] = torch.rand(n, device=self.device)

        self._sample_commands(env_ids)

    def _compute_observations(self) -> TensorDict:
        quat = self.qpos[:, 3:7]
        lin_vel_body = quat_rotate_inverse(quat, self.qvel[:, 0:3])
        # TODO(v5): MuJoCo free-joint qvel[:, 3:6] is already in body frame; fix & retrain in v5.
        ang_vel_body = quat_rotate_inverse(quat, self.qvel[:, 3:6])
        proj_grav = quat_rotate_inverse(quat, self.gravity_vec)

        q_act = self.qpos[:, self.act_qpos_idx] - self.default_act_pos
        dq_act = self.qvel[:, self.act_qvel_idx]

        cmd_speed = torch.linalg.norm(self.commands, dim=-1, keepdim=True)
        moving_mask = (cmd_speed > 0.08).to(torch.float32)
        phase_rad = 2.0 * math.pi * self.gait_phase.unsqueeze(-1)
        gait_clock = torch.cat([torch.sin(phase_rad), torch.cos(phase_rad)], dim=-1) * moving_mask

        policy_clean = torch.cat(
            [
                ang_vel_body * 0.25,
                proj_grav,
                self.commands * torch.tensor([1.0, 1.0, 0.5], device=self.device),
                gait_clock,
                q_act,
                dq_act * 0.05,
                self.actions,
            ],
            dim=-1,
        )

        if self.cfg_obj.obs_noise:
            noise_vec = torch.zeros_like(policy_clean)
            noise_vec[:, 0:3] = 0.05  # ang_vel
            noise_vec[:, 3:6] = 0.03  # gravity
            noise_vec[:, 11:32] = 0.015  # joint pos
            noise_vec[:, 32:53] = 0.04  # joint vel
            policy_obs = policy_clean + torch.randn_like(policy_clean) * noise_vec
        else:
            policy_obs = policy_clean

        touch_l = torch.clamp(self.sensordata[:, self.touch_l_adr : self.touch_l_adr + 1] / 400.0, 0.0, 2.0)
        touch_r = torch.clamp(self.sensordata[:, self.touch_r_adr : self.touch_r_adr + 1] / 400.0, 0.0, 2.0)
        pelvis_z = self.qpos[:, 2:3]
        sole_l_z = self.site_xpos[:, self.sole_l_site_id, 2:3]
        sole_r_z = self.site_xpos[:, self.sole_r_site_id, 2:3]

        critic_obs = torch.cat(
            [
                policy_clean,
                lin_vel_body * 0.5,
                touch_l,
                touch_r,
                pelvis_z,
                sole_l_z,
                sole_r_z,
            ],
            dim=-1,
        )

        return TensorDict(
            {"policy": policy_obs, "critic": critic_obs},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._obs_td

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_clipped = torch.clamp(actions.to(self.device), -3.0, 3.0)
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions_clipped)

        # Apply PD target positions
        target_pos = self.default_act_pos.unsqueeze(0) + self.cfg_obj.action_scale * self.actions
        ctrl_buf = self.default_ctrl.unsqueeze(0).repeat(self.num_envs, 1)
        ctrl_buf[:, self.act_ctrl_idx] = target_pos
        ctrl_buf = torch.clamp(ctrl_buf, self.ctrl_low, self.ctrl_high)
        self.ctrl.copy_(ctrl_buf)

        # Record pre-step foot XY for slip penalty
        self.foot_prev_xy[:, 0, :] = self.site_xpos[:, self.sole_l_site_id, :2]
        self.foot_prev_xy[:, 1, :] = self.site_xpos[:, self.sole_r_site_id, :2]

        # Advance physics decimation substeps
        for _ in range(self.cfg_obj.decimation):
            mjw.step(self.mw_model, self.mw_data)
        wp.synchronize()

        self.episode_length_buf += 1

        # Advance gait clock (natural human cadence scaling with speed)
        vx_abs = torch.abs(self.commands[:, 0])
        gait_freq = 0.90 + 0.35 * torch.clamp(vx_abs, 0.0, 2.0)
        self.gait_phase = (self.gait_phase + gait_freq * self.control_dt) % 1.0

        # Periodic command resampling and disturbance pushes
        resample_mask = (self.episode_length_buf > 0) & (self.episode_length_buf % int(5.0 / self.control_dt) == 0)
        if resample_mask.any():
            self._sample_commands(resample_mask.nonzero(as_tuple=False).squeeze(-1))

        if self.cfg_obj.enable_pushes:
            push_every = int(self.cfg_obj.push_interval_s / self.control_dt)
            push_mask = (self.episode_length_buf > 25) & (self.episode_length_buf % push_every == 0)
            if push_mask.any():
                p_ids = push_mask.nonzero(as_tuple=False).squeeze(-1)
                self.qvel[p_ids, 0:2] += torch.empty((p_ids.shape[0], 2), device=self.device).uniform_(-0.32, 0.32)

        # Compute rewards & terminations before reset
        rewards, reward_log = self._compute_rewards()
        terminated, time_outs = self._check_terminations()
        dones = terminated | time_outs

        # Penalty for early fall termination (not timeout)
        rewards = torch.where(terminated, rewards - 3.0, rewards)

        # Final obs before reset (for RSL-RL)
        final_obs = self._compute_observations()

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._reset_idx(done_ids)
            mjw.kinematics(self.mw_model, self.mw_data)
            mjw.com_pos(self.mw_model, self.mw_data)
            mjw.sensor_pos(self.mw_model, self.mw_data)
            wp.synchronize()
            self.foot_prev_xy[done_ids, 0, :] = self.site_xpos[done_ids, self.sole_l_site_id, :2]
            self.foot_prev_xy[done_ids, 1, :] = self.site_xpos[done_ids, self.sole_r_site_id, :2]

        self._obs_td = self._compute_observations()

        extras = {
            "time_outs": time_outs,
            "final_obs": final_obs,
            "log": reward_log,
        }
        return self._obs_td, rewards, dones, extras

    def _compute_rewards(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        quat = self.qpos[:, 3:7]
        lin_vel_body = quat_rotate_inverse(quat, self.qvel[:, 0:3])
        # TODO(v5): MuJoCo free-joint qvel[:, 3:6] is already in body frame; fix & retrain in v5.
        ang_vel_body = quat_rotate_inverse(quat, self.qvel[:, 3:6])
        proj_grav = quat_rotate_inverse(quat, self.gravity_vec)

        q_act = self.qpos[:, self.act_qpos_idx]
        dq_act = self.qvel[:, self.act_qvel_idx]
        target_pos = torch.clamp(
            self.default_act_pos.unsqueeze(0) + self.cfg_obj.action_scale * self.actions,
            self.ctrl_low[self.act_ctrl_idx],
            self.ctrl_high[self.act_ctrl_idx],
        )
        torques = torch.clamp(
            self.kp_act * (target_pos - q_act) - self.kv_act * dq_act,
            -self.tau_limit_act,
            self.tau_limit_act,
        )

        cmd_speed = torch.linalg.norm(self.commands, dim=-1)
        moving = cmd_speed > 0.08
        standing = ~moving
        moving_f = moving.to(torch.float32)

        # Progress along commanded direction + motion gating so freezing on move commands earns near zero
        vx_err = (lin_vel_body[:, 0] - self.commands[:, 0]) ** 2
        vy_err = (lin_vel_body[:, 1] - self.commands[:, 1]) ** 2
        lin_vel_err = vx_err + 2.5 * vy_err
        cmd_xy_norm = torch.linalg.norm(self.commands[:, :2], dim=-1)
        cmd_dir = self.commands[:, :2] / torch.clamp(cmd_xy_norm.unsqueeze(-1), min=1e-4)
        proj_speed = torch.sum(lin_vel_body[:, :2] * cmd_dir, dim=-1)
        speed_ratio = torch.where(
            cmd_xy_norm > 0.08,
            torch.clamp(proj_speed / torch.clamp(cmd_xy_norm, min=0.35), 0.0, 1.0),
            torch.ones_like(cmd_speed),
        )
        stand_still_vel_rew = (
            torch.exp(-torch.sum(lin_vel_body[:, :2] ** 2, dim=-1) / 0.02) * 4.5
            - torch.sum(lin_vel_body[:, :2] ** 2, dim=-1) * 4.0
        )
        rew_lin_vel = (torch.exp(-lin_vel_err / 0.20) * 4.8 + 1.6 * speed_ratio) * moving_f + stand_still_vel_rew * (1.0 - moving_f)

        motion_gate = torch.where(cmd_xy_norm > 0.08, 0.10 + 0.90 * speed_ratio, torch.ones_like(cmd_speed))

        ang_vel_err = (ang_vel_body[:, 2] - self.commands[:, 2]) ** 2
        rew_ang_vel = torch.exp(-ang_vel_err / 0.10) * 2.8 * motion_gate - ang_vel_err * 2.0

        # Upright pelvis AND upper torso orientation (sharp Gaussian sweet-spot + quadratic guard)
        torso_proj_grav = quat_rotate_inverse(self.xquat[:, self.torso_body_id, :], self.gravity_vec)
        lean_sin = (0.09 if self.cfg_obj.task == "run" else 0.035) * moving_f
        torso_err_sq = (torso_proj_grav[:, 0] - lean_sin) ** 2 + 1.5 * (torso_proj_grav[:, 1] ** 2)
        rew_orient = (
            ((proj_grav[:, 0] - 0.5 * lean_sin) ** 2) * -3.5
            + (proj_grav[:, 1] ** 2) * -4.0
            + torso_err_sq * -4.5
            + torch.exp(-torso_err_sq / 0.018) * 1.8 * motion_gate
        )
        target_z = torch.where(moving, 0.98 if self.cfg_obj.task == "run" else 1.015, 1.036)
        rew_height = ((self.qpos[:, 2] - target_z) ** 2) * -4.0

        # Smooth base velocity
        vz_weight = -0.35 if self.cfg_obj.task == "run" else -1.0
        rew_lin_z = (lin_vel_body[:, 2] ** 2) * vz_weight
        rew_ang_xy = torch.sum(ang_vel_body[:, :2] ** 2, dim=-1) * -0.12

        # Action smoothness & energy
        rew_action_rate = torch.sum((self.actions - self.last_actions) ** 2, dim=-1) * -0.02
        rew_dof_vel = torch.sum(dq_act ** 2, dim=-1) * -1.2e-4
        rew_torque = torch.sum(torques ** 2, dim=-1) * -5e-6

        # Soft joint limit avoidance
        margin = 0.05 * (self.jnt_high - self.jnt_low)
        out_low = torch.clamp((self.jnt_low + margin) - q_act, min=0.0)
        out_high = torch.clamp(q_act - (self.jnt_high - margin), min=0.0)
        rew_limits = torch.sum(out_low + out_high, dim=-1) * -2.0

        # Spine stillness, straight feet (zero pigeon/duck twist or splay), and ankle roll alignment
        is_run = self.cfg_obj.task == "run"
        spine_flex_tgt = (0.055 if is_run else 0.02) * moving_f + 0.02 * (1.0 - moving_f)
        rew_spine_reg = (
            ((q_act[:, 0] - spine_flex_tgt) ** 2) * -5.5
            + (q_act[:, 1] ** 2) * -4.5
            + (q_act[:, 2] ** 2) * -6.0
        )
        # Keep hip_twist (5,11), ankle_abd (8,14), hip_abd (4,10) dead-straight so shins swing straight forward
        twist_sq = q_act[:, 5] ** 2 + q_act[:, 11] ** 2
        ank_abd_sq = q_act[:, 8] ** 2 + q_act[:, 14] ** 2
        hip_abd_sq = (q_act[:, 4] + 0.01) ** 2 + (q_act[:, 10] + 0.01) ** 2
        sole_l_rel_y = quat_rotate_inverse(quat, self.site_xpos[:, self.sole_l_site_id, :] - self.qpos[:, 0:3])[:, 1]
        sole_r_rel_y = quat_rotate_inverse(quat, self.site_xpos[:, self.sole_r_site_id, :] - self.qpos[:, 0:3])[:, 1]
        foot_track_y_sq = (sole_l_rel_y - 0.18) ** 2 + (sole_r_rel_y + 0.18) ** 2
        rew_leg_align = (
            twist_sq * -16.0
            + ank_abd_sq * -18.0
            + hip_abd_sq * -9.0
            + foot_track_y_sq * -7.0
            + torch.exp(-twist_sq / 0.006) * 0.9 * motion_gate
            + torch.exp(-ank_abd_sq / 0.005) * 0.9 * motion_gate
        )

        # Natural contra-lateral arm swing (athletic bent elbows when running, relaxed elbows when walking)
        hip_diff = q_act[:, 3] - q_act[:, 9]
        arm_gain = 0.75 if is_run else 0.60
        arm_lim = 0.65 if is_run else 0.40
        sh_l_tgt = torch.clamp(-arm_gain * hip_diff, -arm_lim, arm_lim) * moving_f
        sh_r_tgt = torch.clamp(+arm_gain * hip_diff, -arm_lim, arm_lim) * moving_f
        elbow_base = (0.95 if is_run else 0.30) * moving_f + 0.30 * (1.0 - moving_f)
        el_l_tgt = elbow_base + 0.25 * torch.clamp(sh_l_tgt, min=0.0)
        el_r_tgt = elbow_base + 0.25 * torch.clamp(sh_r_tgt, min=0.0)
        rew_arm_swing = (
            ((q_act[:, 15] - sh_l_tgt) ** 2 + (q_act[:, 18] - sh_r_tgt) ** 2) * -1.1
            + ((q_act[:, 15] + q_act[:, 18]) ** 2) * -0.8
            + ((q_act[:, 17] - el_l_tgt) ** 2 + (q_act[:, 20] - el_r_tgt) ** 2) * -0.7
            + ((q_act[:, 16] - 0.04) ** 2 + (q_act[:, 19] - 0.04) ** 2) * -2.8
        )

        # Foot contacts & alternating human gait schedule (56% walk vs 42% run with aerial flight phase)
        touch_l = self.sensordata[:, self.touch_l_adr] > 25.0
        touch_r = self.sensordata[:, self.touch_r_adr] > 25.0
        contacts = torch.stack([touch_l, touch_r], dim=-1)
        contact_filt = contacts | self.last_contacts

        duty = 0.42 if is_run else 0.56
        des_l = (self.gait_phase < duty) | standing
        des_r = (((self.gait_phase + 0.5) % 1.0) < duty) | standing
        exact_match = ((contacts[:, 0] == des_l) & (contacts[:, 1] == des_r)).to(torch.float32)
        if is_run:
            # Penalize any double-support during running AND encourage clean double-flight windows
            both_on_run = (contacts[:, 0] & contacts[:, 1] & moving).to(torch.float32)
            rew_gait_phase = exact_match * 2.0 - both_on_run * 0.65
        else:
            single_support_des = des_l ^ des_r
            both_stuck_ss = (contacts[:, 0] & contacts[:, 1] & single_support_des & moving).to(torch.float32)
            rew_gait_phase = exact_match * 1.6 - both_stuck_ss * 0.45

        # Straight stance knee in walk, springy stance knee in run; cap excessive high-kicks
        stance_knee_lim = 0.40 if is_run else 0.24
        stance_l_f = des_l.to(torch.float32) * moving_f
        stance_r_f = des_r.to(torch.float32) * moving_f
        max_swing_knee = 1.25 if is_run else 0.98
        rew_stance_knee = (
            (torch.clamp(q_act[:, 6] - stance_knee_lim, min=0.0) ** 2 * stance_l_f
             + torch.clamp(q_act[:, 12] - stance_knee_lim, min=0.0) ** 2 * stance_r_f) * -2.2
            + (torch.clamp(q_act[:, 6] - max_swing_knee, min=0.0) ** 2
               + torch.clamp(q_act[:, 12] - max_swing_knee, min=0.0) ** 2) * -3.5
        )

        # Level foot sole lateral roll in BOTH stance and swing so lifted foot never tilts sideways
        foot_l_grav = quat_rotate_inverse(self.xquat[:, self.foot_l_body_id, :], self.gravity_vec)
        foot_r_grav = quat_rotate_inverse(self.xquat[:, self.foot_r_body_id, :], self.gravity_vec)
        rew_foot_flat = (foot_l_grav[:, 1] ** 2 + foot_r_grav[:, 1] ** 2) * -6.0

        # Air time reward on touchdown
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.control_dt
        air_gain = 2.8 if is_run else 2.2
        air_max = 0.34 if is_run else 0.28
        rew_air_time = torch.sum(
            torch.clamp(self.feet_air_time - 0.06, min=0.0, max=air_max) * first_contact.to(torch.float32),
            dim=-1,
        ) * air_gain * moving_f
        self.feet_air_time *= ~contact_filt
        self.last_contacts.copy_(contacts)

        # Foot clearance during desired swing phase + penalize excessive high-kicking
        sole_z = torch.stack(
            [self.site_xpos[:, self.sole_l_site_id, 2], self.site_xpos[:, self.sole_r_site_id, 2]],
            dim=-1,
        )
        swing_des = torch.stack([~des_l, ~des_r], dim=-1).to(torch.float32)
        target_clr = 0.12 if is_run else 0.075
        max_clr = 0.24 if is_run else 0.13
        rew_clearance = (
            torch.sum(torch.clamp((sole_z - 0.01) / target_clr, 0.0, 1.0) * swing_des, dim=-1) * 1.4
            - torch.sum(torch.clamp(sole_z - max_clr, min=0.0) ** 2, dim=-1) * 8.0
        ) * moving_f

        # Foot slip penalty during stance
        foot_cur_xy = torch.stack(
            [self.site_xpos[:, self.sole_l_site_id, :2], self.site_xpos[:, self.sole_r_site_id, :2]],
            dim=1,
        )
        foot_vel_xy = (foot_cur_xy - self.foot_prev_xy) / self.control_dt
        slip_sq = torch.sum(foot_vel_xy ** 2, dim=-1) * contacts.to(torch.float32)
        rew_slip = torch.sum(slip_sq, dim=-1) * -0.25

        # Stand still regularization toward tall natural posture
        rew_stand = torch.sum(torch.abs(q_act - self.natural_stand_pos), dim=-1) * standing.to(torch.float32) * -1.4

        rew_alive = torch.full((self.num_envs,), 0.20, device=self.device)

        raw_rate = (
            rew_lin_vel
            + rew_ang_vel
            + rew_orient
            + rew_height
            + rew_lin_z
            + rew_ang_xy
            + rew_action_rate
            + rew_dof_vel
            + rew_torque
            + rew_limits
            + rew_spine_reg
            + rew_leg_align
            + rew_arm_swing
            + rew_gait_phase
            + rew_stance_knee
            + rew_foot_flat
            + rew_air_time
            + rew_clearance
            + rew_slip
            + rew_stand
            + rew_alive
        )
        total_reward = torch.clamp(raw_rate, min=0.0) * self.control_dt

        log = {
            "Episode/rew_lin_vel": rew_lin_vel.mean() * self.control_dt,
            "Episode/rew_ang_vel": rew_ang_vel.mean() * self.control_dt,
            "Episode/rew_gait_phase": rew_gait_phase.mean() * self.control_dt,
            "Episode/rew_air_time": rew_air_time.mean() * self.control_dt,
            "Episode/flight_frac": (~contacts[:, 0] & ~contacts[:, 1]).to(torch.float32).mean(),
            "Episode/rew_orient": rew_orient.mean() * self.control_dt,
            "Episode/torso_sin": torso_proj_grav[:, 0].mean(),
            "Episode/pelvis_z": self.qpos[:, 2].mean(),
            "Episode/vel_x": lin_vel_body[:, 0].mean(),
        }
        return total_reward, log

    def _check_terminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self.qpos[:, 3:7]
        proj_grav = quat_rotate_inverse(quat, self.gravity_vec)

        low_pelvis = self.qpos[:, 2] < 0.65
        tilted = proj_grav[:, 2] > -0.50  # > 60 deg tilt

        # Check non-foot bodies hitting floor (z < 0.14 m)
        non_foot_z = self.xpos[:, self.non_foot_body_ids, 2]
        body_hit_floor = (non_foot_z < 0.14).any(dim=-1)

        nan_guard = ~torch.isfinite(self.qpos).all(dim=-1) | ~torch.isfinite(self.qvel).all(dim=-1)
        terminated = low_pelvis | tilted | body_hit_floor | nan_guard
        time_outs = (self.episode_length_buf >= self.max_episode_length) & (~terminated)
        return terminated, time_outs
