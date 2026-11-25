import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { createMaterialBank } from '../src/config/materials.ts';
import type { PartCatalogEntry } from '../src/config/partsCatalog.ts';
import { RIG_JOINTS, RIG_JOINT_MAP, type Side } from '../src/config/rig.ts';
import { PartRegistry } from '../src/robot/PartRegistry.ts';
import { RobotAssembler } from '../src/robot/RobotAssembler.ts';
import { RobotRig } from '../src/robot/RobotRig.ts';
import { registerArmBuilders } from '../src/robot/parts/ArmBuilder.ts';
import { registerHeadBuilders } from '../src/robot/parts/HeadBuilder.ts';
import { registerLegBuilders } from '../src/robot/parts/LegBuilder.ts';
import { registerTorsoBuilders } from '../src/robot/parts/TorsoBuilder.ts';

type Vec3 = readonly [number, number, number];

type ColliderSpec =
  | {
      name: string;
      type: 'box';
      pos: Vec3;
      size: Vec3;
      rgba?: string;
      condim?: number;
      friction?: string;
    }
  | {
      name: string;
      type: 'sphere';
      pos: Vec3;
      radius: number;
      rgba?: string;
      condim?: number;
      friction?: string;
    }
  | {
      name: string;
      type: 'capsule' | 'cylinder';
      fromto: readonly [number, number, number, number, number, number];
      radius: number;
      rgba?: string;
      condim?: number;
      friction?: string;
    };

interface ServoTuning {
  kp: number;
  kv: number;
  maxTorque: number;
  damping: number;
  armature: number;
  frictionloss: number;
}

interface BodySpec {
  bodyName: string;
  jointId: string;
  parentBody: string | null;
  colliders: ColliderSpec[];
  sites?: Array<{
    name: string;
    type: 'box' | 'sphere';
    pos: Vec3;
    size: string;
    rgba: string;
  }>;
  servo?: Record<'flex' | 'abd' | 'twist', ServoTuning>;
}

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(SCRIPT_DIR, '..');
const ASSET_DIR = resolve(PROJECT_ROOT, 'mujoco', 'humanoid_robot');
const MESH_DIR = resolve(ASSET_DIR, 'meshes');

const DEG2RAD = Math.PI / 180;

/** Maps Three.js [x_left, y_up, z_front] -> MuJoCo [x_front, y_left, z_up] (det = +1). */
function threeToMujoco(v: Vec3): [number, number, number] {
  return [v[2], v[0], v[1]];
}

function mirrorColliderY(col: ColliderSpec, side: Side): ColliderSpec {
  const s = side === 'L' ? 1 : -1;
  const suffix = side;
  if (col.type === 'box') {
    return {
      ...col,
      name: `${col.name}_${suffix}`,
      pos: [col.pos[0], col.pos[1] * s, col.pos[2]],
    };
  }
  if (col.type === 'sphere') {
    return {
      ...col,
      name: `${col.name}_${suffix}`,
      pos: [col.pos[0], col.pos[1] * s, col.pos[2]],
    };
  }
  return {
    ...col,
    name: `${col.name}_${suffix}`,
    fromto: [
      col.fromto[0],
      col.fromto[1] * s,
      col.fromto[2],
      col.fromto[3],
      col.fromto[4] * s,
      col.fromto[5],
    ],
  };
}

function fmt(n: number): string {
  const v = Math.abs(n) < 1e-7 ? 0 : n;
  return Number(v.toFixed(6)).toString();
}

function fmtVec(v: readonly number[]): string {
  return v.map(fmt).join(' ');
}

function colliderVolume(col: ColliderSpec): number {
  if (col.type === 'box') {
    return 8 * col.size[0] * col.size[1] * col.size[2];
  }
  if (col.type === 'sphere') {
    return (4 / 3) * Math.PI * col.radius ** 3;
  }
  const dx = col.fromto[3] - col.fromto[0];
  const dy = col.fromto[4] - col.fromto[1];
  const dz = col.fromto[5] - col.fromto[2];
  const len = Math.hypot(dx, dy, dz);
  if (col.type === 'cylinder') {
    return Math.PI * col.radius ** 2 * len;
  }
  return Math.PI * col.radius ** 2 * len + (4 / 3) * Math.PI * col.radius ** 3;
}

const JOINT_TO_BODY: Record<string, string> = {
  pelvis: 'humanoid_robot',
  spine_lower: 'lumbar_waist',
  spine_upper: 'torso',
  neck: 'neck',
  head: 'head',
  shoulder_L: 'upper_arm_L',
  shoulder_R: 'upper_arm_R',
  pauldron_L: 'upper_arm_L',
  pauldron_R: 'upper_arm_R',
  elbow_L: 'forearm_L',
  elbow_R: 'forearm_R',
  wrist_L: 'hand_L',
  wrist_R: 'hand_R',
  hip_L: 'thigh_L',
  hip_R: 'thigh_R',
  patella_L: 'thigh_L',
  patella_R: 'thigh_R',
  knee_L: 'shin_L',
  knee_R: 'shin_R',
  ankle_L: 'foot_L',
  ankle_R: 'foot_R',
  toe_L: 'foot_L',
  toe_R: 'foot_R',
};

function buildBodySpecs(): BodySpec[] {
  const specs: BodySpec[] = [
    {
      bodyName: 'humanoid_robot',
      jointId: 'pelvis',
      parentBody: null,
      colliders: [
        { name: 'pelvis_core_col', type: 'box', pos: [0.005, 0, 0.015], size: [0.075, 0.115, 0.072] },
        { name: 'pelvis_waist_bridge_col', type: 'capsule', fromto: [-0.008, 0, 0.062, -0.010, 0, 0.125], radius: 0.050 },
        { name: 'pelvis_hip_drum_L_col', type: 'cylinder', fromto: [0, 0.084, -0.010, 0, 0.166, -0.010], radius: 0.058 },
        { name: 'pelvis_hip_drum_R_col', type: 'cylinder', fromto: [0, -0.084, -0.010, 0, -0.166, -0.010], radius: 0.058 },
      ],
      sites: [
        { name: 'imu_site', type: 'box', pos: [0.028, 0, 0.058], size: '0.020 0.020 0.015', rgba: '0.95 0.25 0.25 0.40' },
      ],
    },
    {
      bodyName: 'lumbar_waist',
      jointId: 'spine_lower',
      parentBody: 'humanoid_robot',
      colliders: [
        { name: 'lumbar_waist_col', type: 'cylinder', fromto: [0.006, 0, 0.022, 0.006, 0, 0.166], radius: 0.082 },
      ],
      servo: {
        flex: { kp: 240, kv: 18, maxTorque: 180, damping: 2.0, armature: 0.03, frictionloss: 0.1 },
        abd: { kp: 220, kv: 16, maxTorque: 160, damping: 2.0, armature: 0.03, frictionloss: 0.1 },
        twist: { kp: 180, kv: 14, maxTorque: 140, damping: 1.5, armature: 0.02, frictionloss: 0.1 },
      },
    },
    {
      bodyName: 'torso',
      jointId: 'spine_upper',
      parentBody: 'lumbar_waist',
      colliders: [
        { name: 'torso_chest_col', type: 'box', pos: [0.012, 0, 0.095], size: [0.118, 0.160, 0.085] },
        { name: 'torso_shoulder_beam_col', type: 'capsule', fromto: [0, -0.185, 0.115, 0, 0.185, 0.115], radius: 0.046 },
      ],
      servo: {
        flex: { kp: 220, kv: 16, maxTorque: 160, damping: 2.0, armature: 0.03, frictionloss: 0.1 },
        abd: { kp: 180, kv: 14, maxTorque: 140, damping: 1.5, armature: 0.02, frictionloss: 0.1 },
        twist: { kp: 160, kv: 12, maxTorque: 120, damping: 1.5, armature: 0.02, frictionloss: 0.1 },
      },
    },
    {
      bodyName: 'neck',
      jointId: 'neck',
      parentBody: 'torso',
      colliders: [
        { name: 'neck_col', type: 'capsule', fromto: [0, 0, 0.010, 0.004, 0, 0.068], radius: 0.036 },
      ],
      servo: {
        flex: { kp: 30, kv: 3.0, maxTorque: 25, damping: 0.5, armature: 0.01, frictionloss: 0.05 },
        abd: { kp: 25, kv: 2.5, maxTorque: 20, damping: 0.5, armature: 0.01, frictionloss: 0.05 },
        twist: { kp: 25, kv: 2.5, maxTorque: 20, damping: 0.5, armature: 0.01, frictionloss: 0.05 },
      },
    },
    {
      bodyName: 'head',
      jointId: 'head',
      parentBody: 'neck',
      colliders: [
        { name: 'head_cranium_col', type: 'capsule', fromto: [-0.005, 0, 0.065, 0.0, 0, 0.095], radius: 0.080 },
      ],
      servo: {
        flex: { kp: 25, kv: 2.5, maxTorque: 20, damping: 0.4, armature: 0.01, frictionloss: 0.05 },
        abd: { kp: 25, kv: 2.5, maxTorque: 20, damping: 0.4, armature: 0.01, frictionloss: 0.05 },
        twist: { kp: 20, kv: 2.0, maxTorque: 15, damping: 0.4, armature: 0.01, frictionloss: 0.05 },
      },
    },
  ];

  for (const side of ['L', 'R'] as const) {
    specs.push(
      {
        bodyName: `upper_arm_${side}`,
        jointId: `shoulder_${side}`,
        parentBody: 'torso',
        colliders: [
          mirrorColliderY({ name: 'shoulder_hub', type: 'sphere', pos: [0, 0, 0], radius: 0.048 }, side),
          mirrorColliderY({ name: 'upper_arm', type: 'capsule', fromto: [0, 0.005, -0.025, -0.008, 0.048, -0.205], radius: 0.046 }, side),
        ],
        servo: {
          flex: { kp: 90, kv: 8.0, maxTorque: 120, damping: 1.0, armature: 0.02, frictionloss: 0.08 },
          abd: { kp: 90, kv: 8.0, maxTorque: 120, damping: 1.0, armature: 0.02, frictionloss: 0.08 },
          twist: { kp: 60, kv: 6.0, maxTorque: 80, damping: 0.8, armature: 0.015, frictionloss: 0.06 },
        },
      },
      {
        bodyName: `forearm_${side}`,
        jointId: `elbow_${side}`,
        parentBody: `upper_arm_${side}`,
        colliders: [
          mirrorColliderY({ name: 'elbow_drum', type: 'cylinder', fromto: [0, -0.034, 0, 0, 0.034, 0], radius: 0.038 }, side),
          mirrorColliderY({ name: 'forearm', type: 'capsule', fromto: [0, 0, -0.022, 0.018, 0.026, -0.250], radius: 0.040 }, side),
        ],
        servo: {
          flex: { kp: 65, kv: 6.0, maxTorque: 80, damping: 0.8, armature: 0.015, frictionloss: 0.05 },
          abd: { kp: 40, kv: 4.0, maxTorque: 40, damping: 0.5, armature: 0.01, frictionloss: 0.05 },
          twist: { kp: 40, kv: 4.0, maxTorque: 40, damping: 0.5, armature: 0.01, frictionloss: 0.05 },
        },
      },
      {
        bodyName: `hand_${side}`,
        jointId: `wrist_${side}`,
        parentBody: `forearm_${side}`,
        colliders: [
          mirrorColliderY({ name: 'hand', type: 'box', pos: [0.004, 0.002, -0.082], size: [0.038, 0.018, 0.082] }, side),
        ],
        servo: {
          flex: { kp: 20, kv: 2.0, maxTorque: 25, damping: 0.3, armature: 0.008, frictionloss: 0.03 },
          abd: { kp: 20, kv: 2.0, maxTorque: 25, damping: 0.3, armature: 0.008, frictionloss: 0.03 },
          twist: { kp: 15, kv: 1.5, maxTorque: 20, damping: 0.3, armature: 0.008, frictionloss: 0.03 },
        },
      },
      {
        bodyName: `thigh_${side}`,
        jointId: `hip_${side}`,
        parentBody: 'humanoid_robot',
        colliders: [
          mirrorColliderY({ name: 'thigh', type: 'capsule', fromto: [0.002, 0.012, -0.050, 0.010, 0.035, -0.320], radius: 0.082 }, side),
        ],
        servo: {
          flex: { kp: 380, kv: 34, maxTorque: 220, damping: 2.5, armature: 0.04, frictionloss: 0.1 },
          abd: { kp: 320, kv: 28, maxTorque: 220, damping: 2.5, armature: 0.04, frictionloss: 0.1 },
          twist: { kp: 200, kv: 18, maxTorque: 160, damping: 1.8, armature: 0.025, frictionloss: 0.1 },
        },
      },
      {
        bodyName: `shin_${side}`,
        jointId: `knee_${side}`,
        parentBody: `thigh_${side}`,
        colliders: [
          mirrorColliderY({ name: 'knee_drum', type: 'cylinder', fromto: [0, -0.057, 0, 0, 0.057, 0], radius: 0.056 }, side),
          mirrorColliderY({ name: 'shin', type: 'capsule', fromto: [0, 0.008, -0.060, -0.028, 0.036, -0.458], radius: 0.062 }, side),
        ],
        servo: {
          flex: { kp: 420, kv: 36, maxTorque: 280, damping: 2.5, armature: 0.04, frictionloss: 0.1 },
          abd: { kp: 120, kv: 10, maxTorque: 100, damping: 1.0, armature: 0.02, frictionloss: 0.08 },
          twist: { kp: 120, kv: 10, maxTorque: 100, damping: 1.0, armature: 0.02, frictionloss: 0.08 },
        },
      },
      {
        bodyName: `foot_${side}`,
        jointId: `ankle_${side}`,
        parentBody: `shin_${side}`,
        colliders: [
          mirrorColliderY(
            {
              name: 'foot_sole',
              type: 'box',
              pos: [0.0635, 0, -0.122],
              size: [0.1575, 0.061, 0.018],
              rgba: '0.90 0.80 0.15 0.55',
              condim: 4,
              friction: '1.0 0.02 0.005',
            },
            side,
          ),
          mirrorColliderY(
            {
              name: 'foot_instep',
              type: 'box',
              pos: [0.026, 0, -0.080],
              size: [0.095, 0.054, 0.036],
            },
            side,
          ),
        ],
        sites: [
          {
            name: `foot_${side}_sole_site`,
            type: 'box',
            pos: [0.0635, 0, -0.136],
            size: '0.165 0.068 0.012',
            rgba: '0.15 0.80 0.85 0.35',
          },
        ],
        servo: {
          flex: { kp: 560, kv: 52, maxTorque: 120, damping: 2.0, armature: 0.03, frictionloss: 0.08 },
          abd: { kp: 260, kv: 24, maxTorque: 60, damping: 1.2, armature: 0.02, frictionloss: 0.06 },
          twist: { kp: 60, kv: 6, maxTorque: 40, damping: 0.5, armature: 0.01, frictionloss: 0.05 },
        },
      },
    );
  }

  return specs;
}

interface ObjBucket {
  positions: number[];
  normals: number[];
  uvs: number[];
  faces: Array<[number, number, number]>;
}

function appendGeometryToBucket(
  bucket: ObjBucket,
  geometry: THREE.BufferGeometry,
  meshToBody: THREE.Matrix4,
): void {
  const posAttr = geometry.getAttribute('position');
  if (!posAttr) return;
  const nrmAttr = geometry.getAttribute('normal');
  const uvAttr = geometry.getAttribute('uv');
  const idxAttr = geometry.getIndex();

  const normalMat = new THREE.Matrix3().getNormalMatrix(meshToBody);
  const v = new THREE.Vector3();
  const n = new THREE.Vector3();

  const vertexOffset = bucket.positions.length / 3;
  for (let i = 0; i < posAttr.count; i++) {
    v.fromBufferAttribute(posAttr, i).applyMatrix4(meshToBody);
    const [mx, my, mz] = threeToMujoco([v.x, v.y, v.z]);
    bucket.positions.push(mx, my, mz);

    if (nrmAttr) {
      n.fromBufferAttribute(nrmAttr, i).applyMatrix3(normalMat).normalize();
      const [nx, ny, nz] = threeToMujoco([n.x, n.y, n.z]);
      bucket.normals.push(nx, ny, nz);
    } else {
      bucket.normals.push(0, 0, 1);
    }

    if (uvAttr) {
      bucket.uvs.push(uvAttr.getX(i), uvAttr.getY(i));
    } else {
      bucket.uvs.push(0, 0);
    }
  }

  const triCount = idxAttr ? idxAttr.count / 3 : posAttr.count / 3;
  for (let t = 0; t < triCount; t++) {
    const i0 = idxAttr ? idxAttr.getX(t * 3) : t * 3;
    const i1 = idxAttr ? idxAttr.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = idxAttr ? idxAttr.getX(t * 3 + 2) : t * 3 + 2;
    bucket.faces.push([vertexOffset + i0 + 1, vertexOffset + i1 + 1, vertexOffset + i2 + 1]);
  }
}

function writeObj(filePath: string, bucket: ObjBucket): void {
  const lines: string[] = ['# Exported from humanoid-robot-prototype (Three.js -> MuJoCo +Z up, +X front, +Y left)'];
  for (let i = 0; i < bucket.positions.length; i += 3) {
    lines.push(`v ${fmt(bucket.positions[i]!)} ${fmt(bucket.positions[i + 1]!)} ${fmt(bucket.positions[i + 2]!)}`);
  }
  for (let i = 0; i < bucket.uvs.length; i += 2) {
    lines.push(`vt ${fmt(bucket.uvs[i]!)} ${fmt(bucket.uvs[i + 1]!)}`);
  }
  for (let i = 0; i < bucket.normals.length; i += 3) {
    lines.push(`vn ${fmt(bucket.normals[i]!)} ${fmt(bucket.normals[i + 1]!)} ${fmt(bucket.normals[i + 2]!)}`);
  }
  for (const [a, b, c] of bucket.faces) {
    lines.push(`f ${a}/${a}/${a} ${b}/${b}/${b} ${c}/${c}/${c}`);
  }
  writeFileSync(filePath, `${lines.join('\n')}\n`, 'utf8');
}

function collectVisualMeshes(
  rig: RobotRig,
  assembler: RobotAssembler,
  bodyNames: ReadonlySet<string>,
): {
  bodyMeshes: Map<string, Map<string, string>>;
  bodyMasses: Map<string, number>;
} {
  const bodyBuckets = new Map<string, Map<string, ObjBucket>>();
  const bodyMasses = new Map<string, number>();

  for (const name of bodyNames) {
    bodyBuckets.set(name, new Map());
    bodyMasses.set(name, 0);
  }

  const invWorldByBody = new Map<string, THREE.Matrix4>();
  for (const [jointId, bodyName] of Object.entries(JOINT_TO_BODY)) {
    if (jointId.startsWith('pauldron_') || jointId.startsWith('patella_') || jointId.startsWith('toe_')) {
      continue;
    }
    const pivot = rig.getPivot(jointId);
    invWorldByBody.set(bodyName, pivot.matrixWorld.clone().invert());
  }

  const getBucket = (bodyName: string, grade: string): ObjBucket => {
    const map = bodyBuckets.get(bodyName)!;
    let bucket = map.get(grade);
    if (!bucket) {
      bucket = { positions: [], normals: [], uvs: [], faces: [] };
      map.set(grade, bucket);
    }
    return bucket;
  };

  const instMat = new THREE.Matrix4();
  const meshToBody = new THREE.Matrix4();

  for (const part of assembler.partsList) {
    const entry: PartCatalogEntry = part.entry;
    const bodyName = JOINT_TO_BODY[entry.parentJoint];
    if (!bodyName) {
      throw new Error(`No MuJoCo body mapping for parentJoint '${entry.parentJoint}'`);
    }

    bodyMasses.set(bodyName, (bodyMasses.get(bodyName) ?? 0) + entry.massKg);
    const invBody = invWorldByBody.get(bodyName)!;

    for (const mesh of part.meshes) {
      const grade = String(mesh.userData.materialGrade ?? 'C');
      const bucket = getBucket(bodyName, grade);

      if (mesh instanceof THREE.InstancedMesh) {
        for (let i = 0; i < mesh.count; i++) {
          mesh.getMatrixAt(i, instMat);
          meshToBody.multiplyMatrices(invBody, mesh.matrixWorld).multiply(instMat);
          appendGeometryToBucket(bucket, mesh.geometry, meshToBody);
        }
      } else {
        meshToBody.multiplyMatrices(invBody, mesh.matrixWorld);
        appendGeometryToBucket(bucket, mesh.geometry, meshToBody);
      }
    }
  }

  const bodyMeshes = new Map<string, Map<string, string>>();
  for (const [bodyName, gradeMap] of bodyBuckets.entries()) {
    const fileMap = new Map<string, string>();
    const grades = [...gradeMap.keys()].sort();
    for (const grade of grades) {
      const bucket = gradeMap.get(grade)!;
      if (bucket.faces.length === 0) continue;
      const filename = `${bodyName}_${grade}.obj`;
      writeObj(resolve(MESH_DIR, filename), bucket);
      fileMap.set(grade, filename);
    }
    bodyMeshes.set(bodyName, fileMap);
  }

  return { bodyMeshes, bodyMasses };
}

function renderBodyXml(
  spec: BodySpec,
  childrenByParent: Map<string, BodySpec[]>,
  bodyMeshes: Map<string, Map<string, string>>,
  bodyMasses: Map<string, number>,
  actuatorsOut: string[],
  excludesOut: string[],
  indent: string,
): string {
  const lines: string[] = [];
  const def = RIG_JOINT_MAP.get(spec.jointId)!;

  let posAttr = 'pos="0 0 0"';
  let extraRootAttr = '';
  if (spec.parentBody === null) {
    extraRootAttr = ' childclass="humanoid_robot" quat="1 0 0 0"';
  } else {
    const parentSpec = spec.parentBody;
    const parentDef = RIG_JOINTS.find((j) => JOINT_TO_BODY[j.id] === parentSpec && !j.id.startsWith('pauldron_') && !j.id.startsWith('patella_') && !j.id.startsWith('toe_'))!;
    const relThree: Vec3 = [
      def.bindWorldPos[0] - parentDef.bindWorldPos[0],
      def.bindWorldPos[1] - parentDef.bindWorldPos[1],
      def.bindWorldPos[2] - parentDef.bindWorldPos[2],
    ];
    posAttr = `pos="${fmtVec(threeToMujoco(relThree))}"`;
    excludesOut.push(`    <exclude name="${parentSpec}_${spec.bodyName}_exclude" body1="${parentSpec}" body2="${spec.bodyName}"/>`);
  }

  lines.push(`${indent}<body name="${spec.bodyName}"${extraRootAttr} ${posAttr}>`);

  // Active joints in exact Three.js 'XZY' intrinsic Euler order: flex (Y), abd (X), twist (Z)
  if (spec.parentBody !== null && spec.servo) {
    const axisSpecs: ReadonlyArray<{
      key: 'flex' | 'abd' | 'twist';
      axis: Vec3;
    }> = [
      { key: 'flex', axis: [0, def.axisSign.flex, 0] },
      { key: 'abd', axis: [def.axisSign.abd, 0, 0] },
      { key: 'twist', axis: [0, 0, def.axisSign.twist] },
    ];

    for (const ax of axisSpecs) {
      const lim = def.limits[ax.key];
      if (!lim) continue;
      const tuning = spec.servo[ax.key];
      const jointName = `${spec.jointId}_${ax.key}`;
      const minRad = lim.min * DEG2RAD;
      const maxRad = lim.max * DEG2RAD;

      lines.push(
        `${indent}  <joint name="${jointName}" type="hinge" axis="${fmtVec(ax.axis)}" range="${fmt(minRad)} ${fmt(maxRad)}" damping="${fmt(tuning.damping)}" armature="${fmt(tuning.armature)}" frictionloss="${fmt(tuning.frictionloss)}"/>`,
      );

      actuatorsOut.push(
        `    <position name="${jointName}_servo" joint="${jointName}" kp="${fmt(tuning.kp)}" kv="${fmt(tuning.kv)}" ctrllimited="true" ctrlrange="${fmt(minRad)} ${fmt(maxRad)}" forcelimited="true" forcerange="${fmt(-tuning.maxTorque)} ${fmt(tuning.maxTorque)}"/>`,
      );
    }
  }

  // Visual meshes (group 2, zero mass)
  const gradeFiles = bodyMeshes.get(spec.bodyName);
  if (gradeFiles) {
    for (const [grade] of gradeFiles.entries()) {
      const meshName = `${spec.bodyName}_${grade}_vis`;
      lines.push(
        `${indent}  <geom name="${spec.bodyName}_${grade}_visual" class="visual" mesh="${meshName}" material="mat_${grade}"/>`,
      );
    }
  }

  // Collision primitives (group 3, volume-proportional mass)
  const totalBodyMass = bodyMasses.get(spec.bodyName) ?? 1.0;
  const vols = spec.colliders.map((c) => colliderVolume(c));
  const totalVol = vols.reduce((a, b) => a + b, 0);

  spec.colliders.forEach((col, idx) => {
    const mass = totalVol > 1e-12 ? (totalBodyMass * vols[idx]!) / totalVol : totalBodyMass / spec.colliders.length;
    const rgbaAttr = col.rgba ? ` rgba="${col.rgba}"` : '';
    const condimAttr = col.condim !== undefined ? ` condim="${col.condim}"` : '';
    const fricAttr = col.friction ? ` friction="${col.friction}"` : '';

    if (col.type === 'box') {
      lines.push(
        `${indent}  <geom name="${col.name}" class="collision" type="box" pos="${fmtVec(col.pos)}" size="${fmtVec(col.size)}" mass="${fmt(mass)}"${condimAttr}${fricAttr}${rgbaAttr}/>`,
      );
    } else if (col.type === 'sphere') {
      lines.push(
        `${indent}  <geom name="${col.name}" class="collision" type="sphere" pos="${fmtVec(col.pos)}" size="${fmt(col.radius)}" mass="${fmt(mass)}"${condimAttr}${fricAttr}${rgbaAttr}/>`,
      );
    } else {
      lines.push(
        `${indent}  <geom name="${col.name}" class="collision" type="${col.type}" fromto="${fmtVec(col.fromto)}" size="${fmt(col.radius)}" mass="${fmt(mass)}"${condimAttr}${fricAttr}${rgbaAttr}/>`,
      );
    }
  });

  if (spec.sites) {
    for (const site of spec.sites) {
      lines.push(
        `${indent}  <site name="${site.name}" group="4" type="${site.type}" pos="${fmtVec(site.pos)}" size="${site.size}" rgba="${site.rgba}"/>`,
      );
    }
  }

  const children = childrenByParent.get(spec.bodyName) ?? [];
  for (const child of children) {
    lines.push(
      renderBodyXml(child, childrenByParent, bodyMeshes, bodyMasses, actuatorsOut, excludesOut, `${indent}  `),
    );
  }

  lines.push(`${indent}</body>`);
  return lines.join('\n');
}

function buildModelXml(
  specs: BodySpec[],
  bodyMeshes: Map<string, Map<string, string>>,
  bodyMasses: Map<string, number>,
): string {
  const childrenByParent = new Map<string, BodySpec[]>();
  let rootSpec: BodySpec | null = null;

  for (const spec of specs) {
    if (spec.parentBody === null) {
      rootSpec = spec;
    } else {
      let arr = childrenByParent.get(spec.parentBody);
      if (!arr) {
        arr = [];
        childrenByParent.set(spec.parentBody, arr);
      }
      arr.push(spec);
    }
  }
  if (!rootSpec) throw new Error('Missing root body spec');

  const meshAssets: string[] = [];
  for (const spec of specs) {
    const gradeFiles = bodyMeshes.get(spec.bodyName);
    if (!gradeFiles) continue;
    for (const [grade, filename] of gradeFiles.entries()) {
      meshAssets.push(`    <mesh name="${spec.bodyName}_${grade}_vis" file="${filename}" inertia="shell"/>`);
    }
  }

  const actuators: string[] = [];
  const excludes: string[] = [];
  const worldbodyXml = renderBodyXml(rootSpec, childrenByParent, bodyMeshes, bodyMasses, actuators, excludes, '    ');

  const materialsXml = [
    '    <material name="mat_A" rgba="0.902 0.925 0.961 1.0" metallic="0.10" roughness="0.30" specular="0.60"/>',
    '    <material name="mat_B" rgba="0.725 0.769 0.824 1.0" metallic="0.28" roughness="0.36" specular="0.35"/>',
    '    <material name="mat_C" rgba="0.114 0.133 0.173 1.0" metallic="0.80" roughness="0.42" specular="0.30"/>',
    '    <material name="mat_D" rgba="0.024 0.035 0.063 1.0" metallic="0.15" roughness="0.06" specular="0.90"/>',
    '    <material name="mat_E" rgba="0.306 0.647 0.949 1.0" metallic="0.00" roughness="0.20" emission="2.0"/>',
    '    <material name="mat_F" rgba="0.231 0.267 0.329 1.0" metallic="0.85" roughness="0.30" specular="0.40"/>',
    '    <material name="mat_copper" rgba="0.831 0.541 0.290 1.0" metallic="0.88" roughness="0.28" specular="0.50"/>',
    '    <material name="mat_chrome" rgba="0.847 0.878 0.922 1.0" metallic="0.94" roughness="0.16" specular="0.75"/>',
  ].join('\n');

  return [
    '<mujoco model="humanoid_robot">',
    '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
    '',
    '  <default>',
    '    <default class="humanoid_robot">',
    '      <joint damping="1.0" armature="0.02" frictionloss="0.1" solreflimit="0.005 1"/>',
    '      <default class="visual">',
    '        <geom group="2" type="mesh" contype="0" conaffinity="0" density="0" mass="0"/>',
    '      </default>',
    '      <default class="collision">',
    '        <geom group="3" contype="1" conaffinity="1" condim="4" friction="0.85 0.02 0.005" solref="0.005 1" solimp="0.95 0.99 0.001" margin="0" rgba="0.20 0.75 0.35 0.45"/>',
    '      </default>',
    '    </default>',
    '  </default>',
    '',
    '  <asset>',
    materialsXml,
    ...meshAssets,
    '  </asset>',
    '',
    '  <worldbody>',
    worldbodyXml,
    '  </worldbody>',
    '',
    '  <contact>',
    ...excludes,
    '  </contact>',
    '',
    '  <actuator>',
    ...actuators,
    '  </actuator>',
    '',
    '  <sensor>',
    '    <framequat name="pelvis_quat" objtype="site" objname="imu_site"/>',
    '    <gyro name="pelvis_gyro" site="imu_site"/>',
    '    <accelerometer name="pelvis_acc" site="imu_site"/>',
    '    <subtreecom name="robot_com" body="humanoid_robot"/>',
    '    <touch name="foot_L_touch" site="foot_L_sole_site"/>',
    '    <touch name="foot_R_touch" site="foot_R_sole_site"/>',
    '  </sensor>',
    '</mujoco>',
    '',
  ].join('\n');
}

function main(): void {
  const rig = new RobotRig();
  const materials = createMaterialBank();
  const registry = new PartRegistry();
  registerHeadBuilders(registry);
  registerTorsoBuilders(registry);
  registerArmBuilders(registry, ['L', 'R']);
  registerLegBuilders(registry, ['L', 'R']);

  const assembler = new RobotAssembler();
  assembler.assemble(rig, registry, materials);
  rig.resetToBindPose();
  rig.updateWorldMatrices();
  rig.syncFingerInstances();

  rmSync(MESH_DIR, { recursive: true, force: true });
  mkdirSync(MESH_DIR, { recursive: true });

  const specs = buildBodySpecs();
  const bodyNames = new Set(specs.map((s) => s.bodyName));
  const { bodyMeshes, bodyMasses } = collectVisualMeshes(rig, assembler, bodyNames);

  const modelXml = buildModelXml(specs, bodyMeshes, bodyMasses);
  writeFileSync(resolve(ASSET_DIR, 'model.xml'), modelXml, 'utf8');

  let meshCount = 0;
  for (const m of bodyMeshes.values()) meshCount += m.size;
  let totalMass = 0;
  for (const m of bodyMasses.values()) totalMass += m;

  materials.dispose();

  console.log(`Exported MuJoCo asset to ${ASSET_DIR}`);
  console.log(`  Bodies: ${specs.length}, Visual OBJ meshes: ${meshCount}, Total mass: ${totalMass.toFixed(2)} kg`);
}

main();
