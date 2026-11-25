# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.2.0", "numpy"]
# ///
"""Build interactive scene.xml via MjSpec and validate Phase 1 MuJoCo physics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
MODEL_XML = ROOT_DIR / "humanoid_robot" / "model.xml"
SCENE_XML = ROOT_DIR / "scene.xml"


def build_scene_spec() -> mujoco.MjSpec:
    """Builds an evaluation scene with floor, studio lights, and 6-DoF floating humanoid."""
    scene = mujoco.MjSpec()
    scene.modelname = "humanoid_robot_scene"
    scene.option.timestep = 0.002
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    scene.visual.global_.offwidth = 1280
    scene.visual.global_.offheight = 1280

    # Studio skybox and checkered grid floor
    sky = scene.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.06, 0.10, 0.17]
    sky.rgb2 = [0.01, 0.02, 0.04]
    sky.width = 512
    sky.height = 512

    grid = scene.add_texture()
    grid.name = "floor_grid"
    grid.type = mujoco.mjtTexture.mjTEXTURE_2D
    grid.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    grid.rgb1 = [0.14, 0.18, 0.24]
    grid.rgb2 = [0.09, 0.12, 0.17]
    grid.markrgb = [0.30, 0.65, 0.95]
    grid.mark = mujoco.mjtMark.mjMARK_EDGE
    grid.width = 512
    grid.height = 512

    floor_mat = scene.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.texrepeat = [12, 12]
    floor_mat.texuniform = True
    floor_mat.reflectance = 0.25
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"

    floor = scene.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0, 0, 0.05]
    floor.material = "floor_mat"
    floor.condim = 4
    floor.friction = [1.0, 0.02, 0.005]

    key_light = scene.worldbody.add_light()
    key_light.name = "key_light"
    key_light.pos = [2.5, -2.0, 3.8]
    key_light.dir = [-0.45, 0.35, -0.82]
    key_light.diffuse = [0.95, 0.96, 1.0]

    fill_light = scene.worldbody.add_light()
    fill_light.name = "fill_light"
    fill_light.pos = [-2.2, 2.2, 2.6]
    fill_light.dir = [0.5, -0.5, -0.7]
    fill_light.diffuse = [0.35, 0.45, 0.65]
    fill_light.castshadow = False

    cam_front = scene.worldbody.add_camera()
    cam_front.name = "front_view"
    cam_front.pos = [2.8, 0.0, 0.95]
    cam_front.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam_front.alt.xyaxes = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    cam_side = scene.worldbody.add_camera()
    cam_side.name = "side_view"
    cam_side.pos = [0.0, -2.8, 0.95]
    cam_side.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam_side.alt.xyaxes = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    cam_back = scene.worldbody.add_camera()
    cam_back.name = "back_view"
    cam_back.pos = [-2.8, 0.0, 0.95]
    cam_back.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam_back.alt.xyaxes = [0.0, -1.0, 0.0, 0.0, 0.0, 1.0]

    cam_three_quarter = scene.worldbody.add_camera()
    cam_three_quarter.name = "three_quarter"
    cam_three_quarter.pos = [2.2, -1.8, 1.45]
    cam_three_quarter.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam_three_quarter.alt.xyaxes = [0.6292, 0.7772, 0.0, -0.1415, 0.1145, 0.9833]

    robot_spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    robot_body = robot_spec.worldbody.first_body()
    robot_body.add_freejoint(name="floating_base")

    # Follow cameras (trackcom translates with humanoid COM while preserving world orientation)
    follow_specs = [
        ("follow_three_quarter", [2.2, -1.8, 0.40], [0.6292, 0.7772, 0.0, -0.1415, 0.1145, 0.9833]),
        ("follow_side", [0.0, -2.8, -0.10], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ("follow_front", [2.8, 0.0, -0.10], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        ("follow_back", [-2.8, 0.0, -0.10], [0.0, -1.0, 0.0, 0.0, 0.0, 1.0]),
    ]
    for cname, cpos, caxes in follow_specs:
        fcam = robot_body.add_camera()
        fcam.name = cname
        fcam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
        fcam.pos = cpos
        fcam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        fcam.alt.xyaxes = caxes

    spawn_frame = scene.worldbody.add_frame()
    spawn_frame.pos = [0.0, 0.0, 1.050]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scene.attach(robot_spec, frame=spawn_frame, prefix="")

    return scene


def write_standalone_scene_xml(scene: mujoco.MjSpec) -> None:
    """Exports scene.xml with preserved zero visual mass and humanoid_robot/meshes path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        xml_str = scene.to_xml()

    root = ET.fromstring(xml_str)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", "humanoid_robot/meshes")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    # MjSpec.to_xml() omits density="0" / mass="0" on default visual geoms; restore explicitly.
    for default_el in root.findall(".//default"):
        if default_el.get("class") == "visual":
            geom_el = default_el.find("geom")
            if geom_el is not None:
                geom_el.set("density", "0")
                geom_el.set("mass", "0")

    ET.indent(root, space="  ")
    SCENE_XML.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def check_bind_pose_contacts(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    aabb_min_z = np.min(data.geom_xpos[:, 2])
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    non_floor_contacts = []
    floor_contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if c.geom1 == floor_id or c.geom2 == floor_id:
            other = g2 if c.geom1 == floor_id else g1
            floor_contacts.append((other, float(c.dist)))
        else:
            non_floor_contacts.append((g1, g2, float(c.dist)))

    print(f"[1] Bind pose check: ncon={data.ncon}, min_geom_center_z={aabb_min_z:.4f}")
    print(f"    Floor contacts: {floor_contacts}")
    if non_floor_contacts:
        raise RuntimeError(f"Unexpected self-collisions at bind pose: {non_floor_contacts}")


def check_passive_gravity_drop(model: mujoco.MjModel) -> None:
    """Disables actuators and drops robot under gravity to check collision & numerical stability."""
    data = mujoco.MjData(model)
    # Disable position servo torque by zeroing gainprm on runtime copy
    m_passive = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    m_passive.actuator_gainprm[:, :] = 0.0
    m_passive.actuator_biasprm[:, :] = 0.0
    data = mujoco.MjData(m_passive)
    data.qpos[2] = 1.25  # 20 cm drop

    for _ in range(1250):  # 2.5 seconds
        mujoco.mj_step(m_passive, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError("NaN/Inf detected during passive gravity drop!")

    pelvis_z = float(data.qpos[2])
    min_body_z = float(np.min(data.xpos[1:, 2]))
    print(f"[2] Passive drop test (2.5s): OK | final pelvis_z={pelvis_z:.3f}m, min_body_z={min_body_z:.3f}m")
    if pelvis_z > 0.70 or min_body_z < -0.05:
        raise RuntimeError(f"Unexpected passive drop final height: pelvis_z={pelvis_z}, min_body_z={min_body_z}")


def check_pd_stand_and_push(model: mujoco.MjModel) -> None:
    """Runs PD servo hold at ctrl=0 for 5.0s, then applies an external push to verify tipping physics."""
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0

    # 5.0s PD hold without external pushes
    for _ in range(2500):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError("NaN/Inf during PD stand!")

    pelvis_z_hold = float(data.qpos[2])
    touch_l = float(data.sensor("foot_L_touch").data[0])
    touch_r = float(data.sensor("foot_R_touch").data[0])
    print(f"[3a] PD stand (5.0s, ctrl=0): pelvis_z={pelvis_z_hold:.3f}m, touch=({touch_l:.1f}N, {touch_r:.1f}N)")
    if pelvis_z_hold < 0.95 or touch_l < 300.0 or touch_r < 300.0:
        raise RuntimeError(f"Unpushed stand failed: pelvis_z={pelvis_z_hold}, touch=({touch_l}, {touch_r})")

    # Apply a strong forward push (250 N for 0.30s) on torso
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    for step in range(750):
        data.xfrc_applied[torso_id, 0] = 250.0 if step < 150 else 0.0
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError("NaN/Inf during external push test!")

    pelvis_z_pushed = float(data.qpos[2])
    pelvis_x_pushed = float(data.qpos[0])
    print(f"[3b] After 250N torso push (1.5s): pelvis_x={pelvis_x_pushed:.3f}m, pelvis_z={pelvis_z_pushed:.3f}m")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Launch interactive MuJoCo viewer after checks")
    args = parser.parse_args()

    scene_spec = build_scene_spec()
    write_standalone_scene_xml(scene_spec)
    print(f"Generated {SCENE_XML}")

    # Load from disk so we verify the standalone scene.xml path resolution and mass!
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    total_mass = float(model.body_mass.sum())
    print(
        f"Compiled scene OK: nbody={model.nbody}, njnt={model.njnt}, "
        f"nq={model.nq}, nv={model.nv}, nu={model.nu}, total_mass={total_mass:.2f} kg"
    )
    if abs(total_mass - 81.4) > 0.2:
        raise RuntimeError(f"Expected ~81.4 kg total mass, got {total_mass:.2f} kg")

    check_bind_pose_contacts(model)
    check_passive_gravity_drop(model)
    check_pd_stand_and_push(model)

    if args.viewer:
        from mujoco import viewer as mj_viewer
        data = mujoco.MjData(model)
        mj_viewer.launch(model, data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
