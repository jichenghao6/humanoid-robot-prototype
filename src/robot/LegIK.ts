import * as THREE from 'three';
import { FOOT_GEOMETRY_METRICS } from '../config/proportions.ts';
import { RIG_JOINT_MAP, type Side } from '../config/rig.ts';
import type { RobotRig } from './RobotRig.ts';

const RAD2DEG = 180 / Math.PI;
const DEG2RAD = Math.PI / 180;
const MAX_REACH_RATIO = 0.98;

export interface LegIKTarget {
  /** Desired ankle world position (computed from foot sole contact + foot pitch). */
  ankleWorldPos: THREE.Vector3;
  /** Desired foot pitch in degrees (+ = dorsiflexion / toes up, - = plantarflexion / heel up). */
  footPitchDeg: number;
  /** Desired toe bend in degrees (0..40, positive = toes bend upward to remain flat on ground). */
  toeFlexDeg: number;
}

export interface LegIKResult {
  hipFlexDeg: number;
  hipAbdDeg: number;
  hipTwistDeg: number;
  kneeFlexDeg: number;
  ankleFlexDeg: number;
  ankleAbdDeg: number;
  toeFlexDeg: number;
  clamped: boolean;
}

interface LegBindMetrics {
  v1: THREE.Vector3; // knee - hip in bind pose
  v2: THREE.Vector3; // ankle - knee in bind pose
  cConst: number;
  aConst: number;
  bConst: number;
  rConst: number;
  phiConst: number;
  dMax: number;
  dMin: number;
  hipOffsetFromPelvis: THREE.Vector3;
}

// Module-level scratch objects (zero per-frame heap allocation)
const _invPelvisMat = new THREE.Matrix4();
const _targetInPelvis = new THREE.Vector3();
const _pVec = new THREE.Vector3();
const _aVec = new THREE.Vector3();
const _bVec = new THREE.Vector3();
const _cVec = new THREE.Vector3();
const _hipWorldPos = new THREE.Vector3();
const _pelvisQuat = new THREE.Quaternion();
const _pelvisEuler = new THREE.Euler(0, 0, 0, 'XZY');
const _kneeWorldQuat = new THREE.Quaternion();
const _desiredAnkleWorldQuat = new THREE.Quaternion();
const _desiredUpWorld = new THREE.Vector3();
const _desiredUpInKnee = new THREE.Vector3();
const _tmpEuler = new THREE.Euler(0, 0, 0, 'XZY');
const _rootWorldQuat = new THREE.Quaternion();
const _invKneeWorldQuat = new THREE.Quaternion();
const _relAnkleQuat = new THREE.Quaternion();

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

function wrapAngleRad(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

function computeLegBindMetrics(side: Side): LegBindMetrics {
  const pelvis = RIG_JOINT_MAP.get('pelvis')!.bindWorldPos;
  const hip = RIG_JOINT_MAP.get(`hip_${side}`)!.bindWorldPos;
  const knee = RIG_JOINT_MAP.get(`knee_${side}`)!.bindWorldPos;
  const ankle = RIG_JOINT_MAP.get(`ankle_${side}`)!.bindWorldPos;

  const v1 = new THREE.Vector3(knee[0] - hip[0], knee[1] - hip[1], knee[2] - hip[2]);
  const v2 = new THREE.Vector3(ankle[0] - knee[0], ankle[1] - knee[1], ankle[2] - knee[2]);
  const hipOffsetFromPelvis = new THREE.Vector3(
    hip[0] - pelvis[0],
    hip[1] - pelvis[1],
    hip[2] - pelvis[2],
  );

  // |v1 + Rx(alpha) * v2|^2 = cConst + aConst * cos(alpha) + bConst * sin(alpha)
  const cConst = v1.lengthSq() + v2.lengthSq() + 2 * v1.x * v2.x;
  const aConst = 2 * (v1.y * v2.y + v1.z * v2.z);
  const bConst = 2 * (v1.z * v2.y - v1.y * v2.z);
  const rConst = Math.hypot(aConst, bConst);
  const phiConst = Math.atan2(bConst, aConst);

  // Max reach at alpha = 0 (knee.flex >= 0), min reach at alpha = 135 deg
  const dMaxSq = cConst + aConst; // cos(0)=1, sin(0)=0
  const rad135 = 135 * DEG2RAD;
  const dMinSq = cConst + aConst * Math.cos(rad135) + bConst * Math.sin(rad135);

  return {
    v1,
    v2,
    cConst,
    aConst,
    bConst,
    rConst,
    phiConst,
    dMax: Math.sqrt(Math.max(1e-6, dMaxSq)),
    dMin: Math.sqrt(Math.max(1e-6, dMinSq)),
    hipOffsetFromPelvis,
  };
}

/**
 * Computes the ankle world position from a foot sole reference target (X, groundY, Z)
 * and foot pitch angle so the exact contact point (heel for dorsiflexion, ball for plantarflexion,
 * flat sole for pitch = 0) touches groundY without slip.
 */
export function computeAnkleTargetFromSole(
  soleX: number,
  soleY: number,
  soleCenterZ: number,
  footPitchDeg: number,
  outAnklePos: THREE.Vector3,
): THREE.Vector3 {
  const ankleH = FOOT_GEOMETRY_METRICS.ankleHeight;
  const rx = -footPitchDeg * DEG2RAD; // dorsiflexion (+deg) => -X rotation
  const cosR = Math.cos(rx);
  const sinR = Math.sin(rx);

  if (footPitchDeg > 1e-4) {
    // Heel-rocker pivot: heel contact point in ankle-local space is (0, -ankleH, heelOffsetZ)
    const hy = -ankleH;
    const hz = FOOT_GEOMETRY_METRICS.heelOffsetZ;
    const heelWorldZ = soleCenterZ + (hz - FOOT_GEOMETRY_METRICS.soleCenterOffsetZ);
    const rotY = hy * cosR - hz * sinR;
    const rotZ = hy * sinR + hz * cosR;
    outAnklePos.set(soleX, soleY - rotY, heelWorldZ - rotZ);
  } else if (footPitchDeg < -1e-4) {
    // Forefoot-rocker pivot: toe hinge (0, toeHeight - ankleH, ballOffsetZ) rolls over ground while toe stays flat at y = soleY
    const toeH = FOOT_GEOMETRY_METRICS.toeHeight;
    const hy = toeH - ankleH;
    const bz = FOOT_GEOMETRY_METRICS.ballOffsetZ;
    const ballWorldZ = soleCenterZ + (bz - FOOT_GEOMETRY_METRICS.soleCenterOffsetZ);
    const rotY = hy * cosR - bz * sinR;
    const rotZ = hy * sinR + bz * cosR;
    outAnklePos.set(soleX, soleY + toeH - rotY, ballWorldZ - rotZ);
  } else {
    // Flat foot
    outAnklePos.set(
      soleX,
      soleY + ankleH,
      soleCenterZ - FOOT_GEOMETRY_METRICS.soleCenterOffsetZ,
    );
  }

  return outAnklePos;
}

export class LegIKSolver {
  private readonly metrics: Record<Side, LegBindMetrics>;

  constructor() {
    this.metrics = {
      L: computeLegBindMetrics('L'),
      R: computeLegBindMetrics('R'),
    };
  }

  /**
   * Checks both legs against the 98% max-reach threshold.
   * If either target exceeds 0.98 * dMax, returns a negative vertical offset (meters)
   * by which pelvis should be lowered so both targets become reachable without locking the knee.
   */
  public computeRequiredPelvisDrop(
    rig: RobotRig,
    leftAnkleWorld: THREE.Vector3,
    rightAnkleWorld: THREE.Vector3,
  ): number {
    rig.updateWorldMatrices();
    let maxDrop = 0;

    const pairs: readonly [Side, THREE.Vector3][] = [
      ['L', leftAnkleWorld],
      ['R', rightAnkleWorld],
    ];

    for (const [side, targetWorld] of pairs) {
      const m = this.metrics[side];
      const maxReach = m.dMax * MAX_REACH_RATIO;
      rig.getJointWorldPosition(`hip_${side}`, _hipWorldPos);

      const dx = targetWorld.x - _hipWorldPos.x;
      const dy = targetWorld.y - _hipWorldPos.y;
      const dz = targetWorld.z - _hipWorldPos.z;
      const horizSq = dx * dx + dz * dz;
      const distSq = horizSq + dy * dy;

      if (distSq > maxReach * maxReach) {
        const maxVertSq = Math.max(0.01, maxReach * maxReach - horizSq);
        const allowedVert = Math.sqrt(maxVertSq);
        // Current vertical span is |dy| (target is below hip, so dy < 0)
        const currentVert = Math.abs(dy);
        const neededDrop = Math.max(0, currentVert - allowedVert) + 0.002;
        if (neededDrop > maxDrop) {
          maxDrop = neededDrop;
        }
      }
    }

    return -maxDrop;
  }

  /**
   * Analytically solves hip (flex, abd, twist), knee (flex), ankle (flex, abd), and toe (flex)
   * for one leg, writing directly into RobotRig (subject to joint limits).
   */
  public solveLeg(
    rig: RobotRig,
    side: Side,
    target: LegIKTarget,
    outResult: LegIKResult,
  ): LegIKResult {
    const m = this.metrics[side];
    const pelvisPivot = rig.getPivot('pelvis');
    const hipDef = rig.getJointDef(`hip_${side}`);
    const ankleDef = rig.getJointDef(`ankle_${side}`);

    // Ensure pelvis world matrix is up to date
    pelvisPivot.updateWorldMatrix(true, false);

    // Transform desired ankle world position into pelvis local space
    _invPelvisMat.copy(pelvisPivot.matrixWorld).invert();
    _targetInPelvis.copy(target.ankleWorldPos).applyMatrix4(_invPelvisMat);

    // Vector from hip to desired ankle in pelvis space
    _pVec.copy(_targetInPelvis).sub(m.hipOffsetFromPelvis);

    const maxReach = m.dMax * 0.998;
    const minReach = m.dMin * 1.02;
    let dist = _pVec.length();
    let clamped = false;
    if (dist > maxReach) {
      _pVec.multiplyScalar(maxReach / dist);
      dist = maxReach;
      clamped = true;
    } else if (dist < minReach) {
      _pVec.multiplyScalar(minReach / Math.max(1e-6, dist));
      dist = minReach;
      clamped = true;
    }

    // Step 1: Analytical knee flex angle (alpha >= 0 bends shin toward -Z)
    const cosArg = clamp((dist * dist - m.cConst) / m.rConst, -1, 1);
    const alphaRad = clamp(m.phiConst + Math.acos(cosArg), 0, 135 * DEG2RAD);
    const kneeFlexDeg = alphaRad * RAD2DEG;

    // Exact hip-local ankle vector a(alpha) = v1 + Rx(alpha) * v2
    const cosA = Math.cos(alphaRad);
    const sinA = Math.sin(alphaRad);
    _aVec.set(
      m.v1.x + m.v2.x,
      m.v1.y + (m.v2.y * cosA - m.v2.z * sinA),
      m.v1.z + (m.v2.y * sinA + m.v2.z * cosA),
    );

    // Step 2: Counter-rotate hip twist against pelvis yaw so foot & knee track straight forward (+Z)
    _pelvisQuat.copy(pelvisPivot.quaternion);
    _pelvisEuler.setFromQuaternion(_pelvisQuat, 'XZY');
    const desiredRy = clamp(
      -_pelvisEuler.y,
      -(hipDef.limits.twist?.max ?? 30) * DEG2RAD,
      (hipDef.limits.twist?.max ?? 30) * DEG2RAD,
    );

    // b = Ry(desiredRy) * a
    const cosY = Math.cos(desiredRy);
    const sinY = Math.sin(desiredRy);
    _bVec.set(
      _aVec.x * cosY + _aVec.z * sinY,
      _aVec.y,
      -_aVec.x * sinY + _aVec.z * cosY,
    );

    // Step 3: Solve Rz(rz) * b having x = p.x  =>  bx * cos(rz) - by * sin(rz) = px
    const rXY = Math.hypot(_bVec.x, _bVec.y);
    const psi = Math.atan2(-_bVec.y, _bVec.x);
    const rawRz = psi - Math.acos(clamp(_pVec.x / Math.max(1e-6, rXY), -1, 1));
    // Clamp hip.abd BEFORE solving rx so YZ foot placement (height & Z-velocity) stays exact
    const rawAbdDeg = (rawRz * RAD2DEG) * hipDef.axisSign.abd;
    const clampedAbdDeg = hipDef.limits.abd
      ? clamp(rawAbdDeg, hipDef.limits.abd.min, hipDef.limits.abd.max)
      : 0;
    const rz = clampedAbdDeg * hipDef.axisSign.abd * DEG2RAD;

    // c = Rz(rz) * b
    const cosZ = Math.cos(rz);
    const sinZ = Math.sin(rz);
    _cVec.set(
      _bVec.x * cosZ - _bVec.y * sinZ,
      _bVec.x * sinZ + _bVec.y * cosZ,
      _bVec.z,
    );

    // Step 4: Solve Rx(rx) * c = p in YZ plane
    const rx = wrapAngleRad(Math.atan2(_pVec.z, _pVec.y) - Math.atan2(_cVec.z, _cVec.y));

    // Convert raw Euler angles (rx, ry, rz) to semantic degrees using axisSign
    const hipFlexDeg = (rx * RAD2DEG) * hipDef.axisSign.flex;
    const hipAbdDeg = clampedAbdDeg;
    const hipTwistDeg = (desiredRy * RAD2DEG) * hipDef.axisSign.twist;

    const clampedHip = rig.setSemanticAngles(`hip_${side}`, hipFlexDeg, hipAbdDeg, hipTwistDeg);
    const clampedKnee = rig.setSemanticAngles(`knee_${side}`, kneeFlexDeg, 0, 0);

    // Step 5: Solve 2-DOF ankle (flex, abd) to align sole normal (+Y) exactly with desired world pitch & zero roll
    const kneePivot = rig.getPivot(`knee_${side}`);
    kneePivot.updateWorldMatrix(true, false);
    kneePivot.getWorldQuaternion(_kneeWorldQuat);

    rig.root.getWorldQuaternion(_rootWorldQuat);
    _tmpEuler.set(-target.footPitchDeg * DEG2RAD, 0, 0, 'XZY');
    _relAnkleQuat.setFromEuler(_tmpEuler);
    _desiredAnkleWorldQuat.copy(_rootWorldQuat).multiply(_relAnkleQuat);

    // Transform desired sole up-normal into knee local frame: R_x(ax) * R_z(az) * (0,1,0) = (-sin az, cos az cos ax, cos az sin ax)
    _desiredUpWorld.set(0, 1, 0).applyQuaternion(_desiredAnkleWorldQuat);
    _invKneeWorldQuat.copy(_kneeWorldQuat).invert();
    _desiredUpInKnee.copy(_desiredUpWorld).applyQuaternion(_invKneeWorldQuat).normalize();

    const azRad = Math.asin(clamp(-_desiredUpInKnee.x, -1, 1));
    const axRad = Math.atan2(_desiredUpInKnee.z, _desiredUpInKnee.y);

    const ankleFlexDeg = (axRad * RAD2DEG) * ankleDef.axisSign.flex;
    const ankleAbdDeg = (azRad * RAD2DEG) * ankleDef.axisSign.abd;

    const clampedAnkle = rig.setSemanticAngles(`ankle_${side}`, ankleFlexDeg, ankleAbdDeg, 0);
    const clampedToe = rig.setSemanticAngles(`toe_${side}`, target.toeFlexDeg, 0, 0);

    outResult.hipFlexDeg = clampedHip.flex;
    outResult.hipAbdDeg = clampedHip.abd;
    outResult.hipTwistDeg = clampedHip.twist;
    outResult.kneeFlexDeg = clampedKnee.flex;
    outResult.ankleFlexDeg = clampedAnkle.flex;
    outResult.ankleAbdDeg = clampedAnkle.abd;
    outResult.toeFlexDeg = clampedToe.flex;
    outResult.clamped = clamped;

    return outResult;
  }
}
