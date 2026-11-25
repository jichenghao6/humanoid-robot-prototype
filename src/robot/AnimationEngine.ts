import * as THREE from 'three';
import { GAIT_PARAMS, type GaitParams } from '../config/gait.ts';
import { FOOT_GEOMETRY_METRICS } from '../config/proportions.ts';
import { RIG_JOINTS, RIG_JOINT_MAP, type Side } from '../config/rig.ts';
import type { MotionMode } from '../core/AppState.ts';
import {
  LegIKSolver,
  computeAnkleTargetFromSole,
  type LegIKResult,
  type LegIKTarget,
} from './LegIK.ts';
import type { RobotRig } from './RobotRig.ts';

const CROSSFADE_DURATION = 0.35;
const TWO_PI = Math.PI * 2;
const JUMP_CYCLE_DURATION = 2.4;
const WAVE_CYCLE_DURATION = 3.0;

export interface FootTrajectorySample {
  inStance: boolean;
  soleX: number;
  soleY: number;
  soleCenterZ: number;
  footPitchDeg: number;
  toeFlexDeg: number;
}

// Module-level scratch objects (zero allocations in RAF)
const _leftSample: FootTrajectorySample = {
  inStance: true,
  soleX: 0,
  soleY: 0,
  soleCenterZ: 0,
  footPitchDeg: 0,
  toeFlexDeg: 0,
};
const _rightSample: FootTrajectorySample = {
  inStance: true,
  soleX: 0,
  soleY: 0,
  soleCenterZ: 0,
  footPitchDeg: 0,
  toeFlexDeg: 0,
};
const _leftTarget: LegIKTarget = {
  ankleWorldPos: new THREE.Vector3(),
  footPitchDeg: 0,
  toeFlexDeg: 0,
};
const _rightTarget: LegIKTarget = {
  ankleWorldPos: new THREE.Vector3(),
  footPitchDeg: 0,
  toeFlexDeg: 0,
};
const _ikResultL: LegIKResult = {
  hipFlexDeg: 0,
  hipAbdDeg: 0,
  hipTwistDeg: 0,
  kneeFlexDeg: 0,
  ankleFlexDeg: 0,
  ankleAbdDeg: 0,
  toeFlexDeg: 0,
  clamped: false,
};
const _ikResultR: LegIKResult = {
  hipFlexDeg: 0,
  hipAbdDeg: 0,
  hipTwistDeg: 0,
  kneeFlexDeg: 0,
  ankleFlexDeg: 0,
  ankleAbdDeg: 0,
  toeFlexDeg: 0,
  clamped: false,
};
const _rootPos = new THREE.Vector3();
const _rootQuat = new THREE.Quaternion();
const _pelvisScratch = new THREE.Vector3();
const _targetQuat = new THREE.Quaternion();

function smoothstep01(t: number): number {
  const c = t < 0 ? 0 : t > 1 ? 1 : t;
  return c * c * (3 - 2 * c);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Samples one leg's foot sole trajectory for Walk or Run at normalized phase p ∈ [0, 1).
 * Outputs coordinates in robot_root local space (before root float & turntable yaw).
 */
export function sampleFootTrajectory(
  params: GaitParams,
  side: Side,
  phase: number,
  out: FootTrajectorySample,
): FootTrajectorySample {
  const p = ((phase % 1) + 1) % 1;
  const signX = side === 'L' ? 1 : -1;
  const soleOffsetZ = FOOT_GEOMETRY_METRICS.soleCenterOffsetZ;
  const soleStartZ = params.stanceStartZ + soleOffsetZ;
  const stanceTravel = params.speed * params.dutyFactor * params.cycleDuration;
  const soleEndZ = soleStartZ - stanceTravel;

  out.soleX = params.stepWidthX * signX;

  if (p < params.dutyFactor) {
    // Stance phase: foot moves backward at exact treadmill speed -v with groundY = 0
    const u = p / params.dutyFactor;
    out.inStance = true;
    out.soleY = 0;
    out.soleCenterZ = soleStartZ - stanceTravel * u;

    // Heel-rocker [0..0.12] -> Flat [0.12..0.78] -> Forefoot push-off [0.78..1.0]
    if (u < 0.12) {
      const k = smoothstep01(u / 0.12);
      out.footPitchDeg = lerp(params.heelStrikePitchDeg, 0, k);
      out.toeFlexDeg = 0;
    } else if (u < 0.78) {
      out.footPitchDeg = 0;
      out.toeFlexDeg = 0;
    } else {
      const k = smoothstep01((u - 0.78) / 0.22);
      out.footPitchDeg = lerp(0, params.pushOffPitchDeg, k);
      out.toeFlexDeg = Math.min(params.pushOffToeFlexDeg, Math.max(0, -out.footPitchDeg));
    }
  } else {
    // Swing phase: smooth forward return with clearance arc
    const w = (p - params.dutyFactor) / (1 - params.dutyFactor);
    out.inStance = false;

    const zEase =
      params.swingClearanceY > 0.15
        ? Math.pow(smoothstep01(w), 1.25)
        : smoothstep01(w);
    out.soleCenterZ = lerp(soleEndZ, soleStartZ, zEase);

    const lift = 0.5 * (1 - Math.cos(TWO_PI * w));
    out.soleY = params.swingClearanceY * lift;

    const pitchEase = smoothstep01(w);
    out.footPitchDeg = lerp(params.pushOffPitchDeg, params.heelStrikePitchDeg, pitchEase);
    out.toeFlexDeg = w < 0.35 ? lerp(params.pushOffToeFlexDeg, 0, smoothstep01(w / 0.35)) : 0;
  }

  return out;
}

export class AnimationEngine {
  public readonly legIK = new LegIKSolver();

  private mode: MotionMode = 'Idle';
  private elapsedTime = 0;
  private modeTime = 0;
  private gaitPhase = 0; // Continuous phase ∈ [0, 1) preserved between Walk & Run

  // Cross-fade snapshot state
  private fadeElapsed = CROSSFADE_DURATION;
  private readonly snapshotQuats = new Map<string, THREE.Quaternion>();
  private readonly snapshotPelvisOffset = new THREE.Vector3();

  /** Current treadmill ground speed (m/s) for stage scrolling */
  public treadmillSpeed = 0;
  /** Current LED pulse intensity multiplier (2.5..4.0) */
  public coreEmissiveIntensity = 2.8;

  constructor() {
    for (const def of RIG_JOINTS) {
      if (def.id === 'robot_root') continue;
      this.snapshotQuats.set(def.id, new THREE.Quaternion());
    }
  }

  public getMode(): MotionMode {
    return this.mode;
  }

  public getGaitPhase(): number {
    return this.gaitPhase;
  }

  public setMode(nextMode: MotionMode, rig: RobotRig, immediate = false): void {
    if (nextMode === this.mode && !immediate) return;

    if (!immediate && nextMode !== 'Manual') {
      // Snapshot current joint orientations and pelvis offset for 0.35s Cross-Fade
      rig.getPelvisOffset(this.snapshotPelvisOffset);
      for (const def of RIG_JOINTS) {
        if (def.id === 'robot_root') continue;
        this.snapshotQuats.get(def.id)!.copy(rig.getPivot(def.id).quaternion);
      }
      this.fadeElapsed = 0;
    } else {
      this.fadeElapsed = CROSSFADE_DURATION;
    }

    const switchingBetweenGaits =
      (this.mode === 'Walk' && nextMode === 'Run') ||
      (this.mode === 'Run' && nextMode === 'Walk');
    if (!switchingBetweenGaits && (nextMode === 'Walk' || nextMode === 'Run')) {
      this.gaitPhase = 0;
    }

    this.modeTime = 0;
    this.mode = nextMode;
  }

  /**
   * Evaluates a specific gait phase directly (deterministic helper for tests & sampling).
   */
  public evaluateGaitAtPhase(
    gaitMode: 'Walk' | 'Run',
    phase: number,
    rig: RobotRig,
  ): {
    leftSample: FootTrajectorySample;
    rightSample: FootTrajectorySample;
    ikL: LegIKResult;
    ikR: LegIKResult;
  } {
    this.mode = gaitMode;
    this.gaitPhase = ((phase % 1) + 1) % 1;
    this.fadeElapsed = CROSSFADE_DURATION;
    this.applyGaitPose(GAIT_PARAMS[gaitMode], this.gaitPhase, rig);
    return {
      leftSample: _leftSample,
      rightSample: _rightSample,
      ikL: _ikResultL,
      ikR: _ikResultR,
    };
  }

  public update(dt: number, speedMultiplier: number, playing: boolean, rig: RobotRig): void {
    const scaledDt = playing ? dt * speedMultiplier : 0;
    this.elapsedTime += scaledDt;
    this.modeTime += scaledDt;

    if (this.mode === 'Manual') {
      // Manual mode preserves current pose & evaluates follower constraints cleanly
      this.treadmillSpeed = 0;
      this.coreEmissiveIntensity = 3.0;
      rig.evaluateFollowers();
      return;
    }

    // Advance continuous gait phase for Walk / Run
    if (this.mode === 'Walk' || this.mode === 'Run') {
      const params = GAIT_PARAMS[this.mode];
      this.gaitPhase = (this.gaitPhase + scaledDt / params.cycleDuration) % 1;
      this.treadmillSpeed = playing ? params.speed * speedMultiplier : 0;
    } else {
      this.treadmillSpeed = 0;
    }

    // 1. Evaluate target procedural pose
    if (this.mode === 'Idle') {
      this.applyIdlePose(this.elapsedTime, rig);
    } else if (this.mode === 'Walk') {
      this.applyGaitPose(GAIT_PARAMS.Walk, this.gaitPhase, rig);
      this.coreEmissiveIntensity = 3.0;
    } else if (this.mode === 'Run') {
      this.applyGaitPose(GAIT_PARAMS.Run, this.gaitPhase, rig);
      this.coreEmissiveIntensity = 3.6 + 0.4 * Math.sin(this.gaitPhase * TWO_PI * 2);
    } else if (this.mode === 'Jump') {
      this.applyJumpPose(this.modeTime % JUMP_CYCLE_DURATION, rig);
    } else if (this.mode === 'Wave') {
      this.applyWavePose(this.modeTime, rig);
    }

    // 2. Apply Cross-Fade blending (0.35s quaternion slerp + pelvis lerp)
    if (this.fadeElapsed < CROSSFADE_DURATION) {
      this.fadeElapsed = Math.min(CROSSFADE_DURATION, this.fadeElapsed + dt);
      const alpha = smoothstep01(this.fadeElapsed / CROSSFADE_DURATION);

      rig.getPelvisOffset(_pelvisScratch);
      rig.setPelvisOffset(
        lerp(this.snapshotPelvisOffset.x, _pelvisScratch.x, alpha),
        lerp(this.snapshotPelvisOffset.y, _pelvisScratch.y, alpha),
        lerp(this.snapshotPelvisOffset.z, _pelvisScratch.z, alpha),
      );

      for (const def of RIG_JOINTS) {
        if (def.id === 'robot_root' || def.category === 'follower') continue;
        const pivot = rig.getPivot(def.id);
        _targetQuat.copy(pivot.quaternion);
        const snap = this.snapshotQuats.get(def.id)!;
        pivot.quaternion.copy(snap).slerp(_targetQuat, alpha);
        if (def.id === 'toe_L') {
          rig.soleRefs.L.toeRefPivot.quaternion.copy(pivot.quaternion);
        } else if (def.id === 'toe_R') {
          rig.soleRefs.R.toeRefPivot.quaternion.copy(pivot.quaternion);
        }
        rig.syncAnglesFromPivot(def.id);
      }
      rig.evaluateFollowers();
    }
  }

  private applyIdlePose(t: number, rig: RobotRig): void {
    rig.resetToBindPose();

    // Breathing cycle 3.2 s
    const breathPhase = (t / 3.2) * TWO_PI;
    const breath = 0.5 + 0.5 * Math.sin(breathPhase);

    this.coreEmissiveIntensity = lerp(2.5, 3.6, breath);

    rig.setPelvisOffset(0, (breath - 0.5) * 0.006, 0);
    rig.setSemanticAngles('spine_lower', (breath - 0.5) * 1.2, 0, 0);
    rig.setSemanticAngles('spine_upper', (breath - 0.5) * 2.0, 0, 0);

    // Subtle periodic head scan every ~6.5 s (±15 deg)
    const scanCycle = t % 6.5;
    let neckYaw = 0;
    if (scanCycle < 2.2) {
      neckYaw = 15 * Math.sin((scanCycle / 2.2) * Math.PI);
    } else if (scanCycle > 3.2 && scanCycle < 5.4) {
      neckYaw = -15 * Math.sin(((scanCycle - 3.2) / 2.2) * Math.PI);
    }
    rig.setSemanticAngles('neck', (breath - 0.5) * 1.0, 0, neckYaw);
    rig.setSemanticAngles('head', -(breath - 0.5) * 0.8, 0, 0);

    rig.setSemanticAngles('shoulder_L', 0, breath * 1.2, 0);
    rig.setSemanticAngles('shoulder_R', 0, breath * 1.2, 0);
    rig.setSemanticAngles('elbow_L', 4, 0, 0);
    rig.setSemanticAngles('elbow_R', 4, 0, 0);

    rig.evaluateFollowers();
  }

  /**
   * Jump cycle (2.4 s, Section 5.2):
   * 0.0–0.6s deep crouch (-0.28m, hip ~80°, knee ~110°, arms -40°) ->
   * 0.6–0.8s explosive push-off ->
   * 0.8–1.25s parabolic flight (+0.25m, arms +150°) ->
   * 1.25–1.6s landing cushion (knee ~70°) ->
   * 1.6–2.4s stand recovery. Feet pinned via IK during all ground-contact phases!
   */
  private applyJumpPose(t: number, rig: RobotRig): void {
    rig.resetToBindPose();

    let pelvisY = 0;
    let pelvisZ = 0;
    let torsoLean = 0;
    let armFlex = 0;
    let armAbd = 4;
    let elbowFlex = 10;
    let fingerCurl = 20;
    let airborne = false;
    let footPitchDeg = 0;
    let toeFlexDeg = 0;

    if (t < 0.60) {
      // 0.00–0.60s: Crouch preparation
      const k = smoothstep01(t / 0.60);
      pelvisY = lerp(0, -0.28, k);
      pelvisZ = lerp(0, -0.05, k);
      torsoLean = lerp(0, 22, k);
      armFlex = lerp(0, -40, k);
      armAbd = lerp(4, 12, k);
      elbowFlex = lerp(8, 36, k);
      fingerCurl = lerp(15, 55, k);
      this.coreEmissiveIntensity = lerp(2.8, 3.8, k);
    } else if (t < 0.80) {
      // 0.60–0.80s: Explosive push-off (extends toward 98% reach on toes)
      const k = smoothstep01((t - 0.60) / 0.20);
      pelvisY = lerp(-0.28, 0.015, k);
      pelvisZ = lerp(-0.05, 0.01, k);
      torsoLean = lerp(22, 4, k);
      armFlex = lerp(-40, 120, k);
      armAbd = lerp(12, 18, k);
      elbowFlex = lerp(36, 22, k);
      fingerCurl = lerp(55, 35, k);
      footPitchDeg = lerp(0, -18, k);
      toeFlexDeg = lerp(0, 18, k);
      this.coreEmissiveIntensity = 4.0;
    } else if (t < 1.25) {
      // 0.80–1.25s: Parabolic flight (0.45s hang time => peak +0.25m above takeoff under g=9.81)
      airborne = true;
      const tau = (t - 0.80) / 0.45;
      const parabola = 4 * tau * (1 - tau); // 0 -> 1 -> 0
      pelvisY = lerp(0.015, -0.02, tau) + 0.25 * parabola;
      pelvisZ = 0.01;
      torsoLean = lerp(4, 8, tau);
      armFlex = tau < 0.5 ? lerp(120, 150, smoothstep01(tau * 2)) : lerp(150, 75, smoothstep01((tau - 0.5) * 2));
      armAbd = lerp(18, 16, tau);
      elbowFlex = lerp(22, 30, tau) + 6 * parabola;
      fingerCurl = lerp(35, 45, tau);
      this.coreEmissiveIntensity = 4.0;

      // Smoothly blend from exact takeoff IK angles -> mid-air tuck -> exact touchdown IK angles
      const tuck = Math.sin(Math.PI * tau);
      const sTau = smoothstep01(tau);
      const hipBase = lerp(15.69, 12.93, sTau);
      const kneeBase = lerp(22.07, 19.99, sTau);
      const ankleBase = lerp(-10.62, 9.06, sTau);
      const hipFlex = hipBase + 22 * tuck;
      const kneeFlex = kneeBase + 40 * tuck;
      const ankleFlex = ankleBase - 14 * tuck;
      const toeAir = lerp(18, 0, smoothstep01(Math.min(1, tau * 2.5)));
      for (const side of ['L', 'R'] as const) {
        rig.setSemanticAngles(`hip_${side}`, hipFlex, 2, 0);
        rig.setSemanticAngles(`knee_${side}`, kneeFlex, 0, 0);
        rig.setSemanticAngles(`ankle_${side}`, ankleFlex, 0, 0);
        rig.setSemanticAngles(`toe_${side}`, toeAir, 0, 0);
      }
    } else if (t < 1.60) {
      // 1.25–1.60s: Landing absorption (knee ~70°)
      const k = smoothstep01((t - 1.25) / 0.35);
      pelvisY = lerp(-0.02, -0.18, k);
      pelvisZ = lerp(0.01, -0.035, k);
      torsoLean = lerp(8, 18, k);
      armFlex = lerp(75, -18, k);
      armAbd = lerp(16, 12, k);
      elbowFlex = lerp(30, 48, k);
      fingerCurl = 45;
      this.coreEmissiveIntensity = lerp(4.0, 3.2, k);
    } else {
      // 1.60–2.40s: Smooth stand recovery (endpoints match t=0.00 identically)
      const k = smoothstep01((t - 1.60) / 0.80);
      pelvisY = lerp(-0.18, 0, k);
      pelvisZ = lerp(-0.035, 0, k);
      torsoLean = lerp(18, 0, k);
      armFlex = lerp(-18, 0, k);
      armAbd = lerp(12, 4, k);
      elbowFlex = lerp(48, 8, k);
      fingerCurl = lerp(45, 15, k);
      this.coreEmissiveIntensity = lerp(3.2, 2.8, k);
    }

    rig.setPelvisOffset(0, pelvisY, pelvisZ);
    rig.setSemanticAngles('pelvis', torsoLean * 0.25, 0, 0);
    rig.setSemanticAngles('spine_lower', torsoLean * 0.45, 0, 0);
    rig.setSemanticAngles('spine_upper', torsoLean * 0.30, 0, 0);
    rig.setSemanticAngles('neck', -torsoLean * 0.45, 0, 0);
    rig.setSemanticAngles('head', -torsoLean * 0.35, 0, 0);

    for (const side of ['L', 'R'] as const) {
      rig.setSemanticAngles(`shoulder_${side}`, armFlex, armAbd, 0);
      rig.setSemanticAngles(`elbow_${side}`, elbowFlex, 0, 0);
      rig.setHandCurl(side, fingerCurl);
    }

    // Ground contact phases: pin both feet cleanly to the floor via 2-bone IK!
    if (!airborne) {
      rig.root.updateMatrixWorld(false);
      rig.root.getWorldPosition(_rootPos);
      rig.root.getWorldQuaternion(_rootQuat);

      const ankleBindL = RIG_JOINT_MAP.get('ankle_L')!.bindWorldPos;
      const soleCenterZ = FOOT_GEOMETRY_METRICS.soleCenterOffsetZ + ankleBindL[2];

      computeAnkleTargetFromSole(ankleBindL[0], 0, soleCenterZ, footPitchDeg, _leftTarget.ankleWorldPos);
      _leftTarget.footPitchDeg = footPitchDeg;
      _leftTarget.toeFlexDeg = toeFlexDeg;
      _leftTarget.ankleWorldPos.applyQuaternion(_rootQuat).add(_rootPos);

      computeAnkleTargetFromSole(-ankleBindL[0], 0, soleCenterZ, footPitchDeg, _rightTarget.ankleWorldPos);
      _rightTarget.footPitchDeg = footPitchDeg;
      _rightTarget.toeFlexDeg = toeFlexDeg;
      _rightTarget.ankleWorldPos.applyQuaternion(_rootQuat).add(_rootPos);

      const dropY = this.legIK.computeRequiredPelvisDrop(
        rig,
        _leftTarget.ankleWorldPos,
        _rightTarget.ankleWorldPos,
      );
      if (dropY < 0) {
        rig.setPelvisOffset(0, pelvisY + dropY, pelvisZ);
      }

      this.legIK.solveLeg(rig, 'L', _leftTarget, _ikResultL);
      this.legIK.solveLeg(rig, 'R', _rightTarget, _ikResultR);
    }

    rig.evaluateFollowers();
  }

  /**
   * Wave greeting cycle (3.0 s breathing rhythm, continuous 1.6 Hz wave, Section 5.2):
   * Right shoulder flex 30° / abd 85°, right elbow 100°;
   * Forearm waves via shoulder_R.twist ±20° at 1.6 Hz, wrist abd ±15°;
   * Right fingers open (curl 10°); head tilts right 8°; left arm in standing pose.
   */
  private applyWavePose(t: number, rig: RobotRig): void {
    rig.resetToBindPose();

    const breath = 0.5 + 0.5 * Math.sin((t / WAVE_CYCLE_DURATION) * TWO_PI);
    this.coreEmissiveIntensity = lerp(2.7, 3.5, breath);

    // Smooth initial raise-in (0..0.45s), then continuous steady wave (no periodic drop snap!)
    const raiseEnv = t < 0.45 ? smoothstep01(t / 0.45) : 1.0;
    const waveBlend = t < 0.30 ? 0 : t < 0.65 ? smoothstep01((t - 0.30) / 0.35) : 1.0;
    const waveOsc = Math.sin(TWO_PI * 1.6 * Math.max(0, t - 0.30));

    rig.setPelvisOffset(0, (breath - 0.5) * 0.005, 0);
    rig.setSemanticAngles('spine_lower', 0, -2.5 * raiseEnv, -4 * raiseEnv);
    rig.setSemanticAngles('spine_upper', 1.5 * breath, 0, -5 * raiseEnv);
    rig.setSemanticAngles('neck', 2, 0, -6 * raiseEnv);
    // Head tilts warmly to the robot's right (+8 deg)
    rig.setSemanticAngles('head', -2, 8 * raiseEnv, 0);

    // Right arm greeting wave
    rig.setSemanticAngles(
      'shoulder_R',
      30 * raiseEnv,
      85 * raiseEnv,
      20 * waveOsc * waveBlend,
    );
    rig.setSemanticAngles('elbow_R', 100 * raiseEnv, 0, 0);
    rig.setSemanticAngles('wrist_R', -8 * raiseEnv, 15 * waveOsc * waveBlend, 0);
    rig.setHandCurl('R', 10);

    // Left arm stays relaxed in reference standing pose
    rig.setSemanticAngles('shoulder_L', 0, 2, 0);
    rig.setSemanticAngles('elbow_L', 6, 0, 0);
    rig.setHandCurl('L', 18);

    rig.evaluateFollowers();
  }

  private applyGaitPose(params: GaitParams, phaseL: number, rig: RobotRig): void {
    const phaseR = (phaseL + 0.5) % 1;

    // 1. Upper body & pelvis FK
    const midStanceOffset = params.dutyFactor * 0.5;
    const bobArg = TWO_PI * 2 * (phaseL - midStanceOffset);
    const pelvisY =
      params.pelvisBaseOffsetY + params.pelvisBobPolarity * params.pelvisBobAmpY * Math.cos(bobArg);

    const cycleArg = TWO_PI * phaseL;
    const pelvisTwist = -params.pelvisTwistDeg * Math.cos(cycleArg);
    const pelvisRoll = -params.pelvisRollDeg * Math.sin(cycleArg);

    rig.setPelvisOffset(0, pelvisY, 0);
    rig.setSemanticAngles('pelvis', params.torsoLeanDeg * 0.25, pelvisRoll, pelvisTwist);

    const spineLowerTwist = -pelvisTwist * params.spineCounterTwistRatio * 0.35;
    const spineUpperTwist = -pelvisTwist * params.spineCounterTwistRatio * 0.65;

    rig.setSemanticAngles('spine_lower', params.torsoLeanDeg * 0.45, -pelvisRoll * 0.6, spineLowerTwist);
    rig.setSemanticAngles('spine_upper', params.torsoLeanDeg * 0.30, 0, spineUpperTwist);
    rig.setSemanticAngles('neck', -params.torsoLeanDeg * 0.4, 0, -pelvisTwist * 0.25);
    rig.setSemanticAngles('head', -params.torsoLeanDeg * 0.35, -pelvisRoll * 0.3, 0);

    // Contralateral arm swing
    const armWaveL = -Math.cos(cycleArg);
    const armWaveR = Math.cos(cycleArg);

    const [sMin, sMax] = params.shoulderFlexRange;
    const [eMin, eMax] = params.elbowFlexRange;
    const armAbd = params.speed > 2.0 ? 7 : 3;

    const sFlexL = lerp(sMin, sMax, 0.5 + 0.5 * armWaveL);
    const sFlexR = lerp(sMin, sMax, 0.5 + 0.5 * armWaveR);
    const eFlexL = lerp(eMin, eMax, 0.5 + 0.5 * Math.cos(cycleArg - Math.PI + 0.35));
    const eFlexR = lerp(eMin, eMax, 0.5 + 0.5 * Math.cos(cycleArg + 0.35));

    rig.setSemanticAngles('shoulder_L', sFlexL, armAbd, 0);
    rig.setSemanticAngles('shoulder_R', sFlexR, armAbd, 0);
    rig.setSemanticAngles('elbow_L', eFlexL, 0, 0);
    rig.setSemanticAngles('elbow_R', eFlexR, 0, 0);
    rig.setSemanticAngles('wrist_L', 0, 0, 0);
    rig.setSemanticAngles('wrist_R', 0, 0, 0);

    const fingerCurl = params.speed > 2.0 ? 55 : 18;
    rig.setHandCurl('L', fingerCurl);
    rig.setHandCurl('R', fingerCurl);

    // 2. Foot trajectories (in robot_root local space) -> transform to world space
    sampleFootTrajectory(params, 'L', phaseL, _leftSample);
    sampleFootTrajectory(params, 'R', phaseR, _rightSample);

    computeAnkleTargetFromSole(
      _leftSample.soleX,
      _leftSample.soleY,
      _leftSample.soleCenterZ,
      _leftSample.footPitchDeg,
      _leftTarget.ankleWorldPos,
    );
    _leftTarget.footPitchDeg = _leftSample.footPitchDeg;
    _leftTarget.toeFlexDeg = _leftSample.toeFlexDeg;

    computeAnkleTargetFromSole(
      _rightSample.soleX,
      _rightSample.soleY,
      _rightSample.soleCenterZ,
      _rightSample.footPitchDeg,
      _rightTarget.ankleWorldPos,
    );
    _rightTarget.footPitchDeg = _rightSample.footPitchDeg;
    _rightTarget.toeFlexDeg = _rightSample.toeFlexDeg;

    rig.root.updateMatrixWorld(false);
    rig.root.getWorldPosition(_rootPos);
    rig.root.getWorldQuaternion(_rootQuat);

    _leftTarget.ankleWorldPos.applyQuaternion(_rootQuat).add(_rootPos);
    _rightTarget.ankleWorldPos.applyQuaternion(_rootQuat).add(_rootPos);

    // 3. Enforce 98% leg reach constraint by lowering pelvis if needed
    const dropY = this.legIK.computeRequiredPelvisDrop(
      rig,
      _leftTarget.ankleWorldPos,
      _rightTarget.ankleWorldPos,
    );
    if (dropY < 0) {
      rig.setPelvisOffset(0, pelvisY + dropY, 0);
    }

    // 4. Solve analytical 2-bone IK for both legs
    this.legIK.solveLeg(rig, 'L', _leftTarget, _ikResultL);
    this.legIK.solveLeg(rig, 'R', _rightTarget, _ikResultR);

    // 5. Evaluate follower joints
    rig.evaluateFollowers();
  }
}
