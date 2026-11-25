import * as THREE from 'three';
import { FOOT_GEOMETRY_METRICS } from '../config/proportions.ts';
import {
  FINGER_CURL_RATIOS,
  FINGER_LENGTHS,
  FINGER_NAMES,
  FINGER_PHALANX_RATIOS,
  FINGER_THUMB_YAW_RAD,
  RIG_JOINTS,
  RIG_JOINT_MAP,
  type FingerName,
  type JointDef,
  type Side,
} from '../config/rig.ts';

export interface SemanticAngles {
  flex: number;
  abd: number;
  twist: number;
}

export interface FingerPhalanxNodes {
  finger: FingerName;
  side: Side;
  jointId: string;
  phalanx1: THREE.Group;
  phalanx2: THREE.Group;
  phalanx3: THREE.Group;
  segLengths: readonly [number, number, number];
  baseRadius: number;
}

export interface SoleRefNodes {
  heel: THREE.Object3D;
  ball: THREE.Object3D;
  center: THREE.Object3D;
  toeRefPivot: THREE.Group;
  tip: THREE.Object3D;
}

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const _euler = new THREE.Euler(0, 0, 0, 'XZY');
const _fingerEuler = new THREE.Euler(0, 0, 0, 'YXZ');
const _identityQuat = new THREE.Quaternion();
const _zAxis = new THREE.Vector3(0, 0, 1);
const _yAxis = new THREE.Vector3(0, 1, 0);
const _tmpQuat = new THREE.Quaternion();
const _thumbYawInvL = new THREE.Quaternion().setFromAxisAngle(_yAxis, FINGER_THUMB_YAW_RAD);
const _thumbYawInvR = new THREE.Quaternion().setFromAxisAngle(_yAxis, -FINGER_THUMB_YAW_RAD);

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

export class RobotRig {
  public readonly root: THREE.Group;
  public readonly pivots = new Map<string, THREE.Group>();
  public readonly bindLocalPos = new Map<string, THREE.Vector3>();
  public readonly angles = new Map<string, SemanticAngles>();
  public readonly soleRefs: Record<Side, SoleRefNodes>;
  public readonly fingerNodes: Record<Side, FingerPhalanxNodes[]>;

  private readonly fingerNodeMap = new Map<string, FingerPhalanxNodes>();
  private readonly fingerDirty: Record<Side, boolean> = { L: true, R: true };
  private fingerSyncHandler: ((side: Side) => void) | null = null;
  private readonly pelvisBindPos = new THREE.Vector3();
  private readonly pelvisOffset = new THREE.Vector3();
  private floatY = 0;
  private turntableYaw = 0;

  constructor() {
    const rootPivot = new THREE.Group();
    rootPivot.name = 'joint_robot_root';
    this.root = rootPivot;
    this.pivots.set('robot_root', rootPivot);
    this.bindLocalPos.set('robot_root', new THREE.Vector3(0, 0, 0));
    this.angles.set('robot_root', { flex: 0, abd: 0, twist: 0 });

    // Create all joint pivots and attach to parent pivots (assembly-exception joints will be reparented to PartContainers by RobotAssembler)
    for (const def of RIG_JOINTS) {
      if (def.id === 'robot_root') continue;

      const pivot = new THREE.Group();
      pivot.name = `joint_${def.id}`;
      this.pivots.set(def.id, pivot);
      this.angles.set(def.id, { flex: 0, abd: 0, twist: 0 });

      const parentDef = def.parent ? RIG_JOINT_MAP.get(def.parent) : undefined;
      const parentPivot = def.parent ? this.pivots.get(def.parent) : undefined;
      if (!parentDef || !parentPivot) {
        throw new Error(`Missing parent joint '${def.parent}' for '${def.id}'`);
      }

      const localPos = new THREE.Vector3(
        def.bindWorldPos[0] - parentDef.bindWorldPos[0],
        def.bindWorldPos[1] - parentDef.bindWorldPos[1],
        def.bindWorldPos[2] - parentDef.bindWorldPos[2],
      );
      this.bindLocalPos.set(def.id, localPos);
      pivot.position.copy(localPos);
      parentPivot.add(pivot);

      if (def.id === 'pelvis') {
        this.pelvisBindPos.copy(localPos);
      }
    }

    // Build skeletal sole reference nodes directly on ankle pivots (never under PartContainers, Section 7.2.3)
    this.soleRefs = {
      L: this.createSoleRefNodes('L'),
      R: this.createSoleRefNodes('R'),
    };

    // Build 3-phalanx finger hierarchies (1.0 / 0.9 / 0.7 curl linkage)
    this.fingerNodes = {
      L: this.createFingerPhalanges('L'),
      R: this.createFingerPhalanges('R'),
    };

    this.resetToBindPose();
    this.root.updateMatrixWorld(true);
  }

  private createFingerPhalanges(side: Side): FingerPhalanxNodes[] {
    const mirrorSign = side === 'L' ? 1 : -1;
    const list: FingerPhalanxNodes[] = [];
    for (const finger of FINGER_NAMES) {
      const jointId = `finger_${finger}_${side}`;
      const phalanx1 = this.getPivot(jointId);
      const totalLen = FINGER_LENGTHS[finger];
      const segLengths: readonly [number, number, number] = [
        totalLen * FINGER_PHALANX_RATIOS[0],
        totalLen * FINGER_PHALANX_RATIOS[1],
        totalLen * FINGER_PHALANX_RATIOS[2],
      ];
      const baseRadius = finger === 'thumb' ? 0.0072 : 0.0058;

      if (finger === 'thumb') {
        phalanx1.quaternion.setFromAxisAngle(_yAxis, -mirrorSign * FINGER_THUMB_YAW_RAD);
      }

      const phalanx2 = new THREE.Group();
      phalanx2.name = `phalanx2_${finger}_${side}`;
      phalanx2.position.set(0, -segLengths[0], 0);
      phalanx1.add(phalanx2);

      const phalanx3 = new THREE.Group();
      phalanx3.name = `phalanx3_${finger}_${side}`;
      phalanx3.position.set(0, -segLengths[1], 0);
      phalanx2.add(phalanx3);

      const node: FingerPhalanxNodes = {
        finger,
        side,
        jointId,
        phalanx1,
        phalanx2,
        phalanx3,
        segLengths,
        baseRadius,
      };
      list.push(node);
      this.fingerNodeMap.set(jointId, node);
    }
    return list;
  }

  public setFingerSyncHandler(handler: ((side: Side) => void) | null): void {
    this.fingerSyncHandler = handler;
    if (handler) {
      this.fingerDirty.L = true;
      this.fingerDirty.R = true;
      this.syncFingerInstances();
    }
  }

  public syncFingerInstances(): void {
    if (!this.fingerSyncHandler) return;
    if (this.fingerDirty.L) {
      this.fingerSyncHandler('L');
      this.fingerDirty.L = false;
    }
    if (this.fingerDirty.R) {
      this.fingerSyncHandler('R');
      this.fingerDirty.R = false;
    }
  }

  private createSoleRefNodes(side: Side): SoleRefNodes {
    const anklePivot = this.getPivot(`ankle_${side}`);
    const ankleH = FOOT_GEOMETRY_METRICS.ankleHeight;

    const heel = new THREE.Object3D();
    heel.name = `sole_heel_${side}`;
    heel.position.set(0, -ankleH, FOOT_GEOMETRY_METRICS.heelOffsetZ);
    anklePivot.add(heel);

    const ball = new THREE.Object3D();
    ball.name = `sole_ball_${side}`;
    ball.position.set(0, -ankleH, FOOT_GEOMETRY_METRICS.ballOffsetZ);
    anklePivot.add(ball);

    const center = new THREE.Object3D();
    center.name = `sole_center_${side}`;
    center.position.set(0, -ankleH, FOOT_GEOMETRY_METRICS.soleCenterOffsetZ);
    anklePivot.add(center);

    const toeRefPivot = new THREE.Group();
    toeRefPivot.name = `sole_toe_ref_${side}`;
    toeRefPivot.position.set(
      0,
      FOOT_GEOMETRY_METRICS.toeHeight - ankleH,
      FOOT_GEOMETRY_METRICS.ballOffsetZ,
    );
    anklePivot.add(toeRefPivot);

    const tip = new THREE.Object3D();
    tip.name = `sole_tip_${side}`;
    tip.position.set(
      0,
      -FOOT_GEOMETRY_METRICS.toeHeight,
      FOOT_GEOMETRY_METRICS.toeTipOffsetFromToeZ,
    );
    toeRefPivot.add(tip);

    return { heel, ball, center, toeRefPivot, tip };
  }

  public getPivot(jointId: string): THREE.Group {
    const pivot = this.pivots.get(jointId);
    if (!pivot) {
      throw new Error(`Unknown jointId: ${jointId}`);
    }
    return pivot;
  }

  public getJointDef(jointId: string): JointDef {
    const def = RIG_JOINT_MAP.get(jointId);
    if (!def) {
      throw new Error(`Unknown joint definition: ${jointId}`);
    }
    return def;
  }

  /**
   * Reparents an assembly-exception joint (fingers or toe) under a PartContainer
   * while preserving its bind offset relative to the parent joint.
   */
  public mountJointOnContainer(jointId: string, container: THREE.Group): void {
    const pivot = this.getPivot(jointId);
    const localPos = this.bindLocalPos.get(jointId);
    if (!localPos) return;
    pivot.position.copy(localPos);
    container.add(pivot);
  }

  public setRootFloatY(floatY: number): void {
    this.floatY = floatY;
    this.root.position.y = this.floatY;
  }

  public getRootFloatY(): number {
    return this.floatY;
  }

  public setTurntableYaw(yawRad: number): void {
    this.turntableYaw = yawRad;
    this.root.rotation.y = this.turntableYaw;
  }

  public setPelvisOffset(dx: number, dy: number, dz: number): void {
    this.pelvisOffset.set(dx, dy, dz);
    const pelvis = this.getPivot('pelvis');
    pelvis.position.set(
      this.pelvisBindPos.x + dx,
      this.pelvisBindPos.y + dy,
      this.pelvisBindPos.z + dz,
    );
  }

  public getPelvisOffset(out: THREE.Vector3): THREE.Vector3 {
    return out.copy(this.pelvisOffset);
  }

  public getPelvisBindHeight(): number {
    return this.pelvisBindPos.y;
  }

  /**
   * Sets semantic angles (in degrees) on a joint, clamping to limits,
   * then converts via axisSign & Euler('XZY') to Quaternion (or Z-curl + 3-phalanx linkage for fingers).
   */
  public setSemanticAngles(
    jointId: string,
    flexDeg: number,
    abdDeg = 0,
    twistDeg = 0,
  ): SemanticAngles {
    const def = this.getJointDef(jointId);
    const state = this.angles.get(jointId)!;

    const cFlex = def.limits.flex
      ? clamp(flexDeg, def.limits.flex.min, def.limits.flex.max)
      : 0;
    const cAbd = def.limits.abd
      ? clamp(abdDeg, def.limits.abd.min, def.limits.abd.max)
      : 0;
    const cTwist = def.limits.twist
      ? clamp(twistDeg, def.limits.twist.min, def.limits.twist.max)
      : 0;

    state.flex = cFlex;
    state.abd = cAbd;
    state.twist = cTwist;

    const pivot = this.getPivot(jointId);
    if (def.category === 'finger') {
      const fNode = this.fingerNodeMap.get(jointId);
      const side = def.side ?? 'L';
      const mirrorSign = side === 'L' ? 1 : -1;
      const baseYaw = fNode?.finger === 'thumb' ? -mirrorSign * FINGER_THUMB_YAW_RAD : 0;
      const curlRad = cFlex * DEG2RAD;
      const rz1 = -mirrorSign * curlRad * FINGER_CURL_RATIOS[0];
      _fingerEuler.set(0, baseYaw, rz1, 'YXZ');
      pivot.quaternion.setFromEuler(_fingerEuler);

      if (fNode) {
        fNode.phalanx2.quaternion.setFromAxisAngle(_zAxis, -mirrorSign * curlRad * FINGER_CURL_RATIOS[1]);
        fNode.phalanx3.quaternion.setFromAxisAngle(_zAxis, -mirrorSign * curlRad * FINGER_CURL_RATIOS[2]);
      }
      this.fingerDirty[side] = true;
      this.syncFingerInstances();
      return state;
    }

    const rx = cFlex * def.axisSign.flex * DEG2RAD;
    const rz = cAbd * def.axisSign.abd * DEG2RAD;
    const ry = cTwist * def.axisSign.twist * DEG2RAD;

    _euler.set(rx, ry, rz, 'XZY');
    pivot.quaternion.setFromEuler(_euler);

    // Mirror toe rotation onto skeletal sole_toe_ref so sole_tip stays accurate without explode offset
    if (jointId === 'toe_L') {
      this.soleRefs.L.toeRefPivot.quaternion.copy(pivot.quaternion);
    } else if (jointId === 'toe_R') {
      this.soleRefs.R.toeRefPivot.quaternion.copy(pivot.quaternion);
    }

    return state;
  }

  /**
   * Synchronizes stored semantic angles (degrees) from the pivot's current quaternion
   * (used during Cross-Fade so Manual sliders & follower constraints stay exact).
   */
  public syncAnglesFromPivot(jointId: string): void {
    const def = this.getJointDef(jointId);
    if (def.id === 'robot_root' || def.category === 'follower') return;
    const state = this.angles.get(jointId)!;
    const pivot = this.getPivot(jointId);

    if (def.category === 'finger') {
      const fNode = this.fingerNodeMap.get(jointId);
      const side = def.side ?? 'L';
      const mirrorSign = side === 'L' ? 1 : -1;
      if (fNode?.finger === 'thumb') {
        const inv = side === 'L' ? _thumbYawInvL : _thumbYawInvR;
        _tmpQuat.copy(inv).multiply(pivot.quaternion);
      } else {
        _tmpQuat.copy(pivot.quaternion);
      }
      const signW = _tmpQuat.w < 0 ? -1 : 1;
      const rz1 = 2 * Math.atan2(_tmpQuat.z * signW, _tmpQuat.w * signW);
      state.flex = clamp((-mirrorSign * rz1) * RAD2DEG, 0, 90);
      state.abd = 0;
      state.twist = 0;
      return;
    }

    _euler.setFromQuaternion(pivot.quaternion, 'XZY');
    state.flex = (_euler.x * RAD2DEG) * def.axisSign.flex;
    state.abd = (_euler.z * RAD2DEG) * def.axisSign.abd;
    state.twist = (_euler.y * RAD2DEG) * def.axisSign.twist;
  }

  /**
   * Convenience: sets a joint quaternion directly from an already-computed quaternion
   * (used during Cross-Fade), and syncs skeletal toe ref if applicable.
   */
  public setJointQuaternion(jointId: string, q: THREE.Quaternion): void {
    const pivot = this.getPivot(jointId);
    pivot.quaternion.copy(q);
    if (jointId === 'toe_L') {
      this.soleRefs.L.toeRefPivot.quaternion.copy(q);
    } else if (jointId === 'toe_R') {
      this.soleRefs.R.toeRefPivot.quaternion.copy(q);
    }
    this.syncAnglesFromPivot(jointId);
  }

  public getSemanticAngles(jointId: string): Readonly<SemanticAngles> {
    const state = this.angles.get(jointId);
    if (!state) {
      throw new Error(`Unknown jointId: ${jointId}`);
    }
    return state;
  }

  /**
   * Evaluates follower constraints:
   * - pauldron_L/R: q = slerp(I, shoulder.q, 0.5)
   * - patella_L/R: flex = 0.5 * knee.flex
   * - finger phalanges: 1.0 / 0.9 / 0.7 curl linkage & instanced mesh sync
   */
  public evaluateFollowers(): void {
    const sides: readonly Side[] = ['L', 'R'];
    for (const side of sides) {
      const shoulder = this.getPivot(`shoulder_${side}`);
      const pauldron = this.getPivot(`pauldron_${side}`);
      pauldron.quaternion.copy(_identityQuat).slerp(shoulder.quaternion, 0.5);

      const kneeAngles = this.getSemanticAngles(`knee_${side}`);
      const patella = this.getPivot(`patella_${side}`);
      const patellaDef = this.getJointDef(`patella_${side}`);
      const patellaFlexRad = 0.5 * kneeAngles.flex * patellaDef.axisSign.flex * DEG2RAD;
      _euler.set(patellaFlexRad, 0, 0, 'XZY');
      patella.quaternion.setFromEuler(_euler);

      // Sync middle & distal phalanges to 0.9 / 0.7 of proximal curl
      const mirrorSign = side === 'L' ? 1 : -1;
      const fingers = this.fingerNodes[side];
      for (let i = 0; i < fingers.length; i++) {
        const fNode = fingers[i]!;
        const curlRad = this.getSemanticAngles(fNode.jointId).flex * DEG2RAD;
        fNode.phalanx2.quaternion.setFromAxisAngle(_zAxis, -mirrorSign * curlRad * FINGER_CURL_RATIOS[1]);
        fNode.phalanx3.quaternion.setFromAxisAngle(_zAxis, -mirrorSign * curlRad * FINGER_CURL_RATIOS[2]);
      }
      this.fingerDirty[side] = true;
    }
    this.syncFingerInstances();
  }

  public setHandCurl(side: Side, curlDeg: number): void {
    for (const finger of FINGER_NAMES) {
      this.setSemanticAngles(`finger_${finger}_${side}`, curlDeg, 0, 0);
    }
  }

  public resetToBindPose(): void {
    this.setPelvisOffset(0, 0, 0);
    for (const def of RIG_JOINTS) {
      if (def.id === 'robot_root') continue;
      const state = this.angles.get(def.id)!;
      state.flex = 0;
      state.abd = 0;
      state.twist = 0;
      const pivot = this.getPivot(def.id);
      if (def.category === 'finger') {
        const fNode = this.fingerNodeMap.get(def.id);
        const mirrorSign = def.side === 'L' ? 1 : -1;
        if (fNode?.finger === 'thumb') {
          pivot.quaternion.setFromAxisAngle(_yAxis, -mirrorSign * FINGER_THUMB_YAW_RAD);
        } else {
          pivot.quaternion.identity();
        }
        if (fNode) {
          fNode.phalanx2.quaternion.identity();
          fNode.phalanx3.quaternion.identity();
        }
      } else {
        pivot.quaternion.identity();
      }
    }
    this.soleRefs.L.toeRefPivot.quaternion.identity();
    this.soleRefs.R.toeRefPivot.quaternion.identity();
    this.evaluateFollowers();
  }

  public updateWorldMatrices(): void {
    this.root.updateMatrixWorld(true);
  }

  public getJointWorldPosition(jointId: string, out: THREE.Vector3): THREE.Vector3 {
    return this.getPivot(jointId).getWorldPosition(out);
  }

  public getFingerTipWorldPosition(
    finger: FingerName,
    side: Side,
    out: THREE.Vector3,
  ): THREE.Vector3 {
    const fNode = this.fingerNodeMap.get(`finger_${finger}_${side}`);
    if (!fNode) {
      throw new Error(`Unknown finger: ${finger}_${side}`);
    }
    return out.set(0, -fNode.segLengths[2], 0).applyMatrix4(fNode.phalanx3.matrixWorld);
  }

  public getLowestSoleY(side: Side, scratch: THREE.Vector3): number {
    const refs = this.soleRefs[side];
    refs.heel.getWorldPosition(scratch);
    let minY = scratch.y;
    refs.ball.getWorldPosition(scratch);
    if (scratch.y < minY) minY = scratch.y;
    refs.tip.getWorldPosition(scratch);
    if (scratch.y < minY) minY = scratch.y;
    return minY;
  }
}
