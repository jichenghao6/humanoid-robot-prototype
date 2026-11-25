export type Side = 'L' | 'R';

export type SemanticAxis = 'flex' | 'abd' | 'twist';

export interface AxisRange {
  min: number;
  max: number;
}

export interface JointLimits {
  flex?: AxisRange;
  abd?: AxisRange;
  twist?: AxisRange;
}

export interface AxisSigns {
  flex: 1 | -1;
  abd: 1 | -1;
  twist: 1 | -1;
}

export type JointCategory =
  | 'root'
  | 'floating_base'
  | 'torso'
  | 'arm'
  | 'leg'
  | 'finger'
  | 'follower';

export interface JointDef {
  id: string;
  parent: string | null;
  /** Optional partId container this joint mounts onto (assembly exception: fingers -> hand_assembly, toe -> foot_assembly). */
  mountPartId?: string;
  /** World bind position [x, y, z] in meters (robot +X left, +Y up, +Z forward). */
  bindWorldPos: readonly [number, number, number];
  side?: Side;
  category: JointCategory;
  limits: JointLimits;
  axisSign: AxisSigns;
  /** Follower constraint target joint ID (for pauldron_* and patella_*). */
  followerOf?: string;
  followerType?: 'pauldron' | 'patella';
}

export const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const;
export type FingerName = (typeof FINGER_NAMES)[number];

export const FINGER_LENGTHS: Record<FingerName, number> = {
  thumb: 0.052,
  index: 0.068,
  middle: 0.074,
  ring: 0.068,
  pinky: 0.056,
};

export const FINGER_PHALANX_RATIOS: readonly [number, number, number] = [0.42, 0.32, 0.26];
/** Joint bend ratios across proximal / middle / distal phalanges (Section 5.1: 1.0 / 0.9 / 0.7). */
export const FINGER_CURL_RATIOS: readonly [number, number, number] = [1.0, 0.9, 0.7];
export const FINGER_THUMB_YAW_RAD = 0.48;

const FINGER_LOCAL_OFFSETS_L: Record<FingerName, readonly [number, number, number]> = {
  thumb: [-0.015, -0.065, 0.032],
  index: [0.008, -0.095, 0.024],
  middle: [0.008, -0.100, 0.008],
  ring: [0.006, -0.096, -0.008],
  pinky: [0.004, -0.088, -0.022],
};

function mirrorX(pos: readonly [number, number, number], side: Side): readonly [number, number, number] {
  return side === 'L' ? pos : [-pos[0], pos[1], pos[2]];
}

function buildPairedJoints(): JointDef[] {
  const joints: JointDef[] = [];
  const sides: readonly Side[] = ['L', 'R'];

  for (const side of sides) {
    const mirrorSign: 1 | -1 = side === 'L' ? 1 : -1;
    const wristPos = mirrorX([0.290, 0.970, 0.000], side);
    const anklePos = mirrorX([0.190, 0.140, -0.020], side);

    joints.push(
      {
        id: `shoulder_${side}`,
        parent: 'spine_upper',
        bindWorldPos: mirrorX([0.205, 1.480, -0.010], side),
        side,
        category: 'arm',
        limits: {
          flex: { min: -50, max: 160 },
          abd: { min: -8, max: 120 },
          twist: { min: -60, max: 60 },
        },
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
      },
      {
        id: `pauldron_${side}`,
        parent: 'spine_upper',
        bindWorldPos: mirrorX([0.205, 1.480, -0.010], side),
        side,
        category: 'follower',
        limits: {},
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
        followerOf: `shoulder_${side}`,
        followerType: 'pauldron',
      },
      {
        id: `elbow_${side}`,
        parent: `shoulder_${side}`,
        bindWorldPos: mirrorX([0.260, 1.250, -0.020], side),
        side,
        category: 'arm',
        limits: {
          flex: { min: 0, max: 135 },
        },
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
      },
      {
        id: `wrist_${side}`,
        parent: `elbow_${side}`,
        bindWorldPos: wristPos,
        side,
        category: 'arm',
        limits: {
          flex: { min: -45, max: 45 },
          abd: { min: -25, max: 25 },
        },
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
      },
    );

    for (const finger of FINGER_NAMES) {
      const offL = FINGER_LOCAL_OFFSETS_L[finger];
      const offX = side === 'L' ? offL[0] : -offL[0];
      joints.push({
        id: `finger_${finger}_${side}`,
        parent: `wrist_${side}`,
        mountPartId: `hand_assembly_${side}`,
        bindWorldPos: [wristPos[0] + offX, wristPos[1] + offL[1], wristPos[2] + offL[2]],
        side,
        category: 'finger',
        limits: {
          flex: { min: 0, max: 90 },
        },
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
      });
    }

    joints.push(
      {
        id: `hip_${side}`,
        parent: 'pelvis',
        bindWorldPos: mirrorX([0.110, 1.040, 0.000], side),
        side,
        category: 'leg',
        limits: {
          flex: { min: -35, max: 115 },
          abd: { min: -10, max: 45 },
          twist: { min: -30, max: 30 },
        },
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
      },
      {
        id: `knee_${side}`,
        parent: `hip_${side}`,
        bindWorldPos: mirrorX([0.150, 0.660, 0.010], side),
        side,
        category: 'leg',
        limits: {
          flex: { min: 0, max: 135 },
        },
        axisSign: { flex: 1, abd: mirrorSign, twist: mirrorSign },
      },
      {
        id: `patella_${side}`,
        parent: `hip_${side}`,
        bindWorldPos: mirrorX([0.150, 0.660, 0.010], side),
        side,
        category: 'follower',
        limits: {},
        axisSign: { flex: 1, abd: mirrorSign, twist: mirrorSign },
        followerOf: `knee_${side}`,
        followerType: 'patella',
      },
      {
        id: `ankle_${side}`,
        parent: `knee_${side}`,
        bindWorldPos: anklePos,
        side,
        category: 'leg',
        limits: {
          flex: { min: -35, max: 35 },
          abd: { min: -15, max: 15 },
        },
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
      },
      {
        id: `toe_${side}`,
        parent: `ankle_${side}`,
        mountPartId: `foot_assembly_${side}`,
        bindWorldPos: [anklePos[0], 0.035, anklePos[2] + 0.140],
        side,
        category: 'leg',
        limits: {
          flex: { min: 0, max: 40 },
        },
        axisSign: { flex: -1, abd: mirrorSign, twist: mirrorSign },
      },
    );
  }

  return joints;
}

export const RIG_JOINTS: readonly JointDef[] = [
  {
    id: 'robot_root',
    parent: null,
    bindWorldPos: [0, 0, 0],
    category: 'root',
    limits: {},
    axisSign: { flex: 1, abd: 1, twist: 1 },
  },
  {
    id: 'pelvis',
    parent: 'robot_root',
    bindWorldPos: [0, 1.050, 0],
    category: 'floating_base',
    limits: {
      flex: { min: -45, max: 45 },
      abd: { min: -30, max: 30 },
      twist: { min: -45, max: 45 },
    },
    axisSign: { flex: 1, abd: 1, twist: 1 },
  },
  {
    id: 'spine_lower',
    parent: 'pelvis',
    bindWorldPos: [0, 1.180, -0.010],
    category: 'torso',
    limits: {
      flex: { min: -15, max: 25 },
      abd: { min: -15, max: 15 },
      twist: { min: -25, max: 25 },
    },
    axisSign: { flex: 1, abd: 1, twist: 1 },
  },
  {
    id: 'spine_upper',
    parent: 'spine_lower',
    bindWorldPos: [0, 1.370, -0.010],
    category: 'torso',
    limits: {
      flex: { min: -10, max: 15 },
      twist: { min: -20, max: 20 },
    },
    axisSign: { flex: 1, abd: 1, twist: 1 },
  },
  {
    id: 'neck',
    parent: 'spine_upper',
    bindWorldPos: [0, 1.545, -0.005],
    category: 'torso',
    limits: {
      flex: { min: -20, max: 20 },
      twist: { min: -60, max: 60 },
    },
    axisSign: { flex: 1, abd: 1, twist: 1 },
  },
  {
    id: 'head',
    parent: 'neck',
    bindWorldPos: [0, 1.620, 0],
    category: 'torso',
    limits: {
      flex: { min: -15, max: 20 },
      abd: { min: -20, max: 20 },
    },
    axisSign: { flex: 1, abd: 1, twist: 1 },
  },
  ...buildPairedJoints(),
];

export const RIG_JOINT_MAP: ReadonlyMap<string, JointDef> = new Map(
  RIG_JOINTS.map((j) => [j.id, j]),
);

export interface RigDofStats {
  torsoActiveDof: number;
  armActiveDof: number;
  legActiveDof: number;
  totalActiveDof: number;
  fingerDof: number;
  floatingBaseDof: number;
}

export function computeRigDofStats(): RigDofStats {
  let torsoActiveDof = 0;
  let armActiveDof = 0;
  let legActiveDof = 0;
  let fingerDof = 0;
  const floatingBaseDof = 6;

  for (const joint of RIG_JOINTS) {
    const count =
      (joint.limits.flex ? 1 : 0) +
      (joint.limits.abd ? 1 : 0) +
      (joint.limits.twist ? 1 : 0);
    if (joint.category === 'torso') torsoActiveDof += count;
    else if (joint.category === 'arm') armActiveDof += count;
    else if (joint.category === 'leg') legActiveDof += count;
    else if (joint.category === 'finger') fingerDof += count;
  }

  return {
    torsoActiveDof,
    armActiveDof,
    legActiveDof,
    totalActiveDof: torsoActiveDof + armActiveDof + legActiveDof,
    fingerDof,
    floatingBaseDof,
  };
}
