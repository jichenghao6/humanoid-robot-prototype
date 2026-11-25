# Technical Report: Procedural Humanoid Robot — MJCF Export, Closed-Loop Control & GPU RL Locomotion

This technical report documents the full engineering pipeline from procedural Three.js mechanical design to MuJoCo rigid-body simulation and GPU-accelerated Reinforcement Learning (RL) for bipedal **Walking** and **Running**.

---

## 1. System Architecture

The project bridges a modular Three.js digital prototype (`1.800 m` height, `81.40 kg`, `57` structural parts) into MuJoCo physics and trains GPU-vectorized locomotion policies with **MuJoCo-Warp** and **RSL-RL PPO**.

```mermaid
flowchart LR
    A["Three.js Prototype\n(src/config + src/robot)"] -->|"npm run export:mjcf\n(scripts/export-mjcf.ts)"| B["MJCF Assets\n(mujoco/humanoid_robot/model.xml\n+ 92 .obj meshes)"]
    B -->|"Phase 1 Build & Audit\n(mujoco/verify_mujoco.py)"| C["Evaluation Scene\n(mujoco/scene.xml)"]
    C -->|"Phase 2 Baseline\n(mujoco/phase2_controller.py)"| D["Closed-Loop Balance,\nSquat & Single-Leg Tests"]
    C -->|"Phase 3 GPU VecEnv\n(mujoco/rl/humanoid_env.py)"| E["MuJoCo-Warp + RSL-RL PPO\n(1024 envs on RTX 4070)"]
    E -->|"mujoco/rl/train.py"| F["Pretrained Artifacts\n(latest_walk / latest_run\n.pt & .onnx)"]
    F -->|"mujoco/rl/play.py"| G["CPU MuJoCo Evals &\nSlow-Mo Tracking Viewer"]
```

---

## 2. Mechanical Specifications & Coordinate Export

### 2.1 Structural Assembly & Proportions

Defined in `src/config/proportions.ts`, `src/config/partsCatalog.ts`, and `src/config/rig.ts`, and audited in `tests/robot.test.ts`:

| Metric | Value | Source |
| --- | --- | --- |
| Standing Height | `1.800 m` | `ROBOT_TOTAL_HEIGHT` |
| Total Mass | `81.40 kg` (`79–83 kg` BOM budget) | `model.xml` / `verify_mujoco.py` |
| Part Hierarchy | `57` parts across `35` entry types (`Tier 1/2/3`) | `PARTS_CATALOG` |
| Three.js Active Semantic DOFs | `35` (`9` torso, `12` arm, `14` leg) + `10` finger coupling + `6` floating-base | `computeRigDofStats()` |
| MuJoCo Exported Model | `nbody=18`, `njnt=34`, `nq=40`, `nv=39`, `nu=33` position servos (`toe_L`/`toe_R` merged into rigid feet) | `mujoco/scene.xml` |
| Onboard Sensors (`15`-dim) | `pelvis_quat[4]`, `pelvis_gyro[3]`, `pelvis_acc[3]`, `robot_com[3]`, `foot_L_touch[1]`, `foot_R_touch[1]` | `mujoco/scene.xml` |

Key segment proportions (`src/config/proportions.ts`):
- **Head & Neck**: Head `y=1.605..1.800 m`, Neck `y=1.545..1.605 m`
- **Torso & Pelvis**: Chest `y=1.370..1.560 m` (width `0.400 m`), Abdomen `y=1.175..1.370 m`, Pelvis `y=0.960..1.175 m`
- **Upper Limbs**: Upper arm `y=1.250..1.480 m`, Forearm `y=0.970..1.250 m`, Hand `y=0.800..0.970 m`
- **Lower Limbs**: Thigh `y=0.660..1.040 m`, Shin `y=0.140..0.660 m`, Foot `y=0.000..0.140 m` (length `0.310 m`, sole width `0.130 m`)

### 2.2 Frame Transform & MJCF Generation

`scripts/export-mjcf.ts` maps right-handed Three.js world coordinates into MuJoCo's Z-up robotics convention (`det = +1`):

$$\begin{bmatrix} x_{\text{mj}} \\ y_{\text{mj}} \\ z_{\text{mj}} \end{bmatrix} = \begin{bmatrix} z_{\text{three}} \\ x_{\text{three}} \\ y_{\text{three}} \end{bmatrix}$$

* Visual geometry is exported as `92` per-material `.obj` meshes with `density="0" mass="0"` (`class="visual"`, `contype="0"`, `conaffinity="0"`).
* Collision physics uses clean primitive proxies (`class="collision"`, `box`, `capsule`, `cylinder`) carrying explicit segment masses and `exclude` contact pairs between adjacent links.
* Neutral feet sit laterally at `y = ±0.190 m` (`dy = 0.380 m` between `foot_L_sole_site` and `foot_R_sole_site`), with exact left/right mirrored joint axes verified in `scene.xml`:
  - `hip_{L,R}_flex`: `[0, -1, 0]` (`+` swings forward)
  - `hip_L_abd / hip_R_abd`: `[+1, 0, 0]` / `[-1, 0, 0]` (`+` abducts outward)
  - `hip_L_twist / hip_R_twist`: `[0, 0, +1]` / `[0, 0, -1]` (`+` rotates outward)
  - `knee_{L,R}_flex`: `[0, 1, 0]` (`+` flexes shin backward)
  - `ankle_{L,R}_flex`: `[0, -1, 0]` (`+` dorsiflexion)
  - `ankle_L_abd / ankle_R_abd`: `[+1, 0, 0]` / `[-1, 0, 0]` (`+` lifts outer sole edge)

### 2.3 Evaluation Scene (`mujoco/scene.xml`)

Generated programmatically via `mujoco.MjSpec` in `mujoco/verify_mujoco.py`:
- Checkered floor (`condim=4`, `friction="1 0.02 0.005"`), gradient skybox, key/fill lights.
- `4` static cameras (`front_view`, `side_view`, `back_view`, `three_quarter`) plus `4` COM-tracking cameras (`follow_three_quarter`, `follow_side`, `follow_front`, `follow_back`, `mode="trackcom"`) attached to `humanoid_robot`.

---

## 3. Phase 1 & Phase 2 Verification Baselines

### Phase 1 (`mujoco/verify_mujoco.py`)

Verified static and passive structural validity after scene generation:
1. **Bind-pose clearance**: `ncon=0`, `min_robot_geom_center_z=0.0180` (excluding floor plane; clean ground spawn at `z=1.050 m` with zero initial self-collisions).
2. **Passive drop (`2.5 s` from `z=1.25 m`, actuators zeroed)**: no `NaN/Inf`, settled cleanly at `final pelvis_z=0.443 m`, `min_body_z=0.038 m`.
3. **Neutral PD stand (`5.0 s`, `ctrl=0`) + `250 N` chest push**: held `pelvis_z=1.039 m` with balanced foot loads `(399.3 N, 399.4 N)` before the push, and fell forward stably under a `250 N / 0.30 s` impulse (`pelvis_x=1.383 m`, `pelvis_z=0.099 m`) without numerical blowup.

### Phase 2 (`mujoco/phase2_controller.py`)

Implemented a classical IMU + CoM + joint-space PD controller (`HumanoidPhase2Controller`) and validated four dynamic checks (`rc=0`):
1. **`10 s` Closed-loop balance + `120 N / 90 N` pushes**: `PASS | pelvis_z=1.004 m, pitch=-2.89 deg, roll=-0.01 deg, touch=(399 N, 399 N)`
2. **`6.4 s` Dynamic squat cycles**: `PASS | pelvis_z range=[0.790 m, 1.050 m], min_foot_touch=335 N`
3. **`7.0 s` Single-leg weight shift**: `PASS | right_sole_max_z=0.177 m, right_touch_min=0.0 N, final_pelvis_z=1.012 m`
4. **Extreme ROM self-collision sweep**: `hits=[]` (`0` unexpected self-contacts across full leg/spine/arm ROM sweeps)

---

## 4. Phase 3 GPU Reinforcement Learning Architecture

### 4.1 Environment & Hardware Setup

Trained on a single consumer laptop GPU using zero-copy PyTorch ↔ Warp CUDA buffer aliasing (`wp.to_torch(...)`):

| Component | Configuration / Version |
| --- | --- |
| GPU / Driver | `NVIDIA GeForce RTX 4070 Laptop GPU` (8 GiB), CUDA Toolkit `13.4` / Driver `13.2` |
| Python & Packages | Python `3.12.10`, `torch==2.14.1+cu126`, `mujoco==4.0.1`, `mujoco-warp==4.0.1`, `warp-lang==1.19.0`, `rsl-rl-lib==6.0.0`, `tensordict==0.15.6`, `onnx==1.23.2`, `onnxscript==0.7.2` |
| Parallel Simulation | `1024` environments, `nconmax=16`, `njmax=64` per environment |
| Integrator & Timestep | `mjINT_IMPLICITFAST`, `sim_dt = 0.004 s`, `decimation = 5` (`50 Hz` control, `control_dt = 0.02 s`), `iterations = 8`, `ls_iterations = 8` |
| Throughput | `~14,000–15,900 RL steps/s` (`~70,000–79,500 sim steps/s`, `~1.6–1.9 s` per `24,576`-step PPO iteration) |
| Disturbance Curriculum | Random XY base velocity perturbations every `6.0 s` (`uniform(-0.32, +0.32) m/s`), commands resampled every `5.0 s` |

### 4.2 Observation, Action & Network Spaces

* **Action space (`21`-dim)**: `ACTION_JOINT_NAMES` controls `spine_lower` (`3`), left leg (`6`: `hip_{flex,abd,twist}`, `knee_flex`, `ankle_{flex,abd}`), right leg (`6`), and left/right arms (`6`: `shoulder_{flex,abd}`, `elbow_flex`). Distal joints (`spine_upper`, `neck`, `head`, `shoulder_twist`, `wrist`) hold neutral PD targets (`0`). Actions are clipped to `[-3.0, 3.0]` and scaled by `action_scale = 0.42` around `DEFAULT_ANGLES_RAD`.
* **Actor Observation (`74`-dim `"policy"`, with Gaussian noise during training)**:
  - Pelvis body angular velocity (`3`, `* 0.25`)
  - Projected gravity vector in pelvis frame (`3`)
  - Velocity command `[vx, vy, wz * 0.5]` (`3`)
  - Gait clock `[sin(2πφ), cos(2πφ)] * moving_mask` (`2`), where `gait_freq = 0.90 + 0.35 * clamp(|vx|, 0, 2.0)`
  - Joint position offsets `q - q_default` (`21`)
  - Joint velocities `dq * 0.05` (`21`)
  - Previous actions (`21`)
* **Asymmetric Critic Observation (`82`-dim `"critic"`)**: noise-free `74`-dim policy obs + `lin_vel_body * 0.5` (`3`), scaled left/right touch forces (`2`), `pelvis_z` (`1`), and left/right sole heights (`2`).
* **Policy Architecture (`rsl_rl 6.0.0`)**: `MLPModel([256, 128, 128], activation="elu", obs_normalization=True)` for both actor and critic; PPO (`clip=0.2`, `gamma=0.99`, `lam=0.95`, `5` epochs, `4` mini-batches, adaptive KL schedule `desired_kl=0.012`).

---

## 5. Iterative Biomechanical Tuning Case Studies

Achieving stable **and** natural human-like locomotion required resolving four concrete failure modes observed in rollout telemetry and visual inspection:

### Case Study 1: Eliminating the Standing-Still Local Minimum

* **Symptom**: During cold-start redesign (`v2` stage 1), adding flat upright torso (`+1.8`) and angular velocity (`+2.2`) bonuses caused PPO to freeze in place (`vel_x = 0.0081 m/s`, `ep_len = 824`) because standing still collected `~5.7 / s` without fall risk.
* **Fix**: Replaced flat velocity Gaussian tails with progress-coupled tracking (`1.6 * speed_ratio + 4.8 * exp(-err / 0.20)`), multiplied upright/heading bonuses by `motion_gate = 0.10 + 0.90 * speed_ratio`, reduced alive baseline (`0.80 -> 0.20`), and resumed from the dynamic walking policy (`latest_walk_v1.pt`) using a dual-reference design (`DEFAULT_ANGLES_RAD` preserved for zero-shock obs normalization while rewards pulled toward upright `NATURAL_STAND_RAD`).

### Case Study 2: Resolving `-23.4°` Limbo Lean & Squat Walking (`v1` -> `v3`/`v4`)

| Kinematic Metric (`cmd vx=0.9 m/s`) | Walk `v1` | Walk `v4` (Final) | Root Cause & Fix |
| --- | --- | --- | --- |
| `torso_pitch` mean `[min, max]` | **`-23.4°`** `[-34.9°, -14.0°]` | **`+0.3°`** | Added dual `pelvis` + `torso` projected gravity tracking with Gaussian posture sweet spot + `spine_lower_flex` centering |
| `torso_roll` std | `12.8°` (penguin wobble) | **`4.3°`** | Tightened `spine_lower_abd` (`-4.5`), `spine_lower_twist` (`-6.0`), and torso roll penalty |
| `pelvis_z` mean | `0.969 m` (crouched) | **`1.032 m`** | Raised target walking height `0.99 -> 1.015 m` and added stance-phase knee straightness (`knee <= 0.24 rad`) |
| `knee L/R` min flexion | `25.1° / 25.7°` | **`~4.5° / ~0°`** (v3/v4) | Stance leg extends naturally through push-off |
| Contra-lateral arm-leg correlation | Uncoordinated | **`+0.934`** (`v3: +0.962`) | Added anti-phase target `shoulder_{L,R}_flex = ∓0.60 * (hip_L - hip_R)`, relaxed elbow flexion, and `shoulder_abd` tuck |

**Before (`Walk v1`, `-23.4°` backward torso lean & bent-knee squat):**
![Walk v1 Side Before](images/walk_v1_before_side.png)

**After (`Walk v4`, upright torso `+0.3°`, tall `1.032 m` stride, natural arm swing):**
![Walk v4 Side After](images/walk_v4_side.png)

### Case Study 3: Diagnosing & Fixing Right-Leg Swing Outward Flare (`v3` -> `v4`)

* **Observation**: Front-view viewer inspection showed the right shin and ankle kicking outward when lifting during swing, while the left leg looked clean.
* **Diagnosis**:
  - `scene.xml` legs were verified 100% mirror-symmetric.
  - Frame-by-frame front rollout telemetry of `v3` revealed excessive right knee swing (`max = 80.1°`, `sr_z_max = 0.266 m`) paired with medial hip twist (`hip_R_twist` down to `-8.9°`), hip adduction (`-7.3°`), and swing-phase ankle roll (`ankle_R_abd` `-7.2°` to `-8.8°`, left ankle `-8.0°`) because `rew_foot_flat` had been masked by `contacts` and turned off during swing.
* **Fix**: Applied full-cycle `rew_foot_flat` (both stance and swing), Gaussian + quadratic locks on `hip_twist` (`-16.0`) and `ankle_abd` (`-18.0`), sagittal foot rail tracking (`sole_L/R_y -> ±0.18 m`), capped excessive swing knee kick (`max 0.98 rad` / `max_clr 0.13 m` in walk), and tightened `STAND` zero-velocity penalties.
* **Result**: Right-swing `hip_R_tw` mean improved **`-3.3° -> -0.1°`**, `ank_R_abd` mean **`-6.1° -> -1.4°`**, and `STAND` creep dropped **`0.131 m/s -> 0.0002 m/s`**.

**Before (`Walk v3` Front View — outward lower-leg flare during swing):**
![Walk v3 Front Before](images/walk_v3_before_front.png)

**After (`Walk v4` Front View — straight parallel leg tracks & level ankles):**
![Walk v4 Front After](images/walk_v4_front.png)

### Case Study 4: Walk-to-Run Curriculum Transfer (`latest_run.pt`)

Resuming from `latest_walk.pt` (`v4`) for `240` GPU iterations (`~6.8 min`) with `duty=0.42`, double-support penalty (`-0.65`), `~5°` forward lean target, `target_z=0.98 m`, springy stance knee allowance (`0.40 rad`), and `~54°` elbow flexion baseline produced symmetric, zero-fall running up to `2.34 m/s`:
- `L swing`: `hip_abd=-1.3°`, `hip_tw=-0.7°`, `ank_abd=-0.4°`, `knee_max=+85.6°`
- `R swing`: `hip_abd=-1.7°`, `hip_tw=-0.5°`, `ank_abd=-0.2°`, `knee_max=+88.8°`
- `elbow_L_flex = +62.3°`, `elbow_R_flex = +69.3°`, `torso_pitch = +4.3°`, `roll_std = 2.7°`
- Note on gait phase: at `cmd vx=2.2 m/s`, the policy uses single-support `86.7%`, double-support `11.3%`, and brief double-flight aerial phase `2.0%` (brisk athletic stride; increasing flight duration further is a direct follow-up via higher `both_on_run` penalty and raised `target_z`).

**Run Side & Front Sequences:**
![Run Side](images/run_v1_side.png)
![Run Front](images/run_v1_front.png)

---

## 6. Quantitative Benchmark Results (Post-Final-Edit Verification)

All checks below ran after the final edit to `mujoco/rl/humanoid_env.py` on standard CPU MuJoCo (`play.py`, `rc=0`, **0 falls** across every scenario):

### 6.1 Walk Policy (`mujoco/rl/checkpoints/latest_walk.pt`)

| Scenario | Command `[vx, vy, wz]` | Duration | Actual `mean_vx` | Actual `mean_wz` | Final `(x, y)` | Pelvis `(min_z / final_z)` |
| --- | --- | --- | --- | --- | --- | --- |
| Stand | `[0.0, 0.0, 0.0]` | `8 s` | `0.000 m/s` | `-0.004 rad/s` | `(-0.004, -0.007) m` | `1.021 / 1.032 m` |
| Slow Walk | `[0.55, 0.0, 0.0]` | `10 s` | `0.657 m/s` | `+0.057 rad/s` | `(6.130, 1.114) m` | `1.010 / 1.033 m` |
| Normal Walk | `[0.9, 0.0, 0.0]` | `10 s` | `1.081 m/s` | `+0.045 rad/s` | `(10.184, 1.239) m` | `0.997 / 1.030 m` |
| Fast Walk | `[1.25, 0.0, 0.0]` | `10 s` | `1.365 m/s` | `+0.015 rad/s` | `(13.041, -0.340) m` | `0.978 / 1.023 m` |
| Walk + `150 N` Push | `[0.9, 0.0, 0.0]` (`150 N @ 3.0..3.2s`) | `10 s` | `1.075 m/s` | `+0.013 rad/s` | `(10.336, -0.210) m` | `0.909 / 1.031 m` |
| Turning Walk | `[0.8, 0.0, 0.5]` | `10 s` | `0.978 m/s` | `+0.169 rad/s` | `(5.808, 5.869) m` | `1.003 / 1.049 m` |

### 6.2 Run Policy (`mujoco/rl/checkpoints/latest_run.pt`)

| Scenario | Command `[vx, vy, wz]` | Duration | Actual `mean_vx` | Actual `mean_wz` | Final `(x, y)` | Pelvis `(min_z / final_z)` |
| --- | --- | --- | --- | --- | --- | --- |
| Stand | `[0.0, 0.0, 0.0]` | `8 s` | `0.030 m/s` | `-0.062 rad/s` | `(0.304, 0.122) m` | `0.992 / 1.014 m` |
| Jog | `[1.4, 0.0, 0.0]` | `10 s` | `1.426 m/s` | `-0.022 rad/s` | `(13.567, -1.810) m` | `0.990 / 1.016 m` |
| Run | `[2.0, 0.0, 0.0]` | `10 s` | `2.105 m/s` | `-0.036 rad/s` | `(19.818, -3.048) m` | `0.950 / 0.967 m` |
| Fast Run | `[2.4, 0.0, 0.0]` | `10 s` | `2.341 m/s` | `-0.029 rad/s` | `(22.196, -1.982) m` | `0.931 / 0.938 m` |
| Run + `150 N` Push | `[2.0, 0.0, 0.0]` (`150 N @ 3.0..3.2s`) | `10 s` | `2.142 m/s` | `-0.025 rad/s` | `(20.359, -1.737) m` | `0.926 / 0.960 m` |
| Turning Run | `[1.8, 0.0, 0.5]` | `10 s` | `1.875 m/s` | `+0.081 rad/s` | `(15.345, 8.447) m` | `0.964 / 1.006 m` |

### 6.3 Software, Build & Model Checks

* **ONNX Runtime Contract**: Both `latest_walk.onnx` and `latest_run.onnx` (`280.9 KB` each) pass `onnx.checker.check_model` with input `obs: [1, 74]` and output `actions: [1, 21]`.
* **Web Prototype & TypeScript**: `npm run typecheck`, `npm test` (`16/16` in `tests/robot.test.ts`), and `npm run build` pass cleanly (`rc=0`).

### 6.4 Known Limitations & v5 Roadmap

1. **Free-joint angular velocity frame (`qvel[:, 3:6]`)**: In MuJoCo, `free` joint `qvel[3:6]` is already expressed in the body frame, whereas `humanoid_env.py` and `play.py` currently pass it through `quat_rotate_inverse` (kept as-is for `v4` checkpoint compatibility). Removing this extra rotation and retraining in `v5` will make angular-velocity observations yaw-invariant.
2. **Turning yaw-rate under-response**: Unconditional `hip_twist` (`-16.0`), `spine_lower_twist` (`-6.0`), and foot-track Y penalties restrict yaw turning (`+0.169 rad/s` on Walk `wz=0.5`, `+0.081 rad/s` on Run `wz=0.5`). `v5` will relax twist/track penalties proportionally when `|cmd_wz| > 0` or `|cmd_vy| > 0`.
3. **Speed overshoot (`~+20%` in Walk)**: The clamped `speed_ratio` term inside `rew_lin_vel` and `motion_gate` creates an asymmetric gradient around `cmd_vx`. `v5` will center tracking around the exact commanded velocity and decouple `motion_gate` from `speed_ratio` clipping.
4. **Light hand/forearm-to-thigh brushing**: Tight `shoulder_abd` (`0.04 rad`) causes brief `2–14 mm` hand/forearm-to-thigh grazing contacts during vigorous swing. `v5` will widen neutral shoulder abduction (`0.10–0.14 rad`) and add a non-foot self-contact penalty.
5. **Reset touch-sensor clearing & `njmax`**: `v5` will explicitly zero `sensordata` touch channels on `_reset_idx` and raise `njmax_per_env` from `64` to `96–128` to prevent constraint overflow during early-training falls.
