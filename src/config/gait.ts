export interface GaitParams {
  /** Single-leg full cycle duration T (seconds). */
  cycleDuration: number;
  /** Stance phase fraction (duty factor): Walk 0.62 (double support), Run 0.35 (double flight). */
  dutyFactor: number;
  /** Treadmill speed v (m/s). */
  speed: number;
  /** Ankle Z at touchdown relative to hip (m). */
  stanceStartZ: number;
  /** Ankle Z at liftoff relative to hip (m). */
  stanceEndZ: number;
  /** Half step width (foot center X magnitude, m). */
  stepWidthX: number;
  /** Peak swing clearance height (m). */
  swingClearanceY: number;
  /** Pelvis baseline vertical offset from bind height (m, negative = crouch). */
  pelvisBaseOffsetY: number;
  /** Pelvis vertical bob amplitude (m) at frequency 2/T. */
  pelvisBobAmpY: number;
  /** +1 = highest at mid-stance (Walk), -1 = lowest at mid-stance (Run). */
  pelvisBobPolarity: 1 | -1;
  /** Pelvis yaw twist amplitude (deg). */
  pelvisTwistDeg: number;
  /** Pelvis roll tilt amplitude (deg). */
  pelvisRollDeg: number;
  /** Torso forward lean (deg). */
  torsoLeanDeg: number;
  /** Shoulder flex [min, max] deg (in-phase with contralateral leg). */
  shoulderFlexRange: readonly [number, number];
  /** Elbow flex [min, max] deg. */
  elbowFlexRange: readonly [number, number];
  /** Spine upper counter-twist ratio relative to pelvis twist. */
  spineCounterTwistRatio: number;
  /** Foot pitch at heel strike (deg, positive = dorsiflexion). */
  heelStrikePitchDeg: number;
  /** Foot pitch at push-off (deg, negative = plantarflexion). */
  pushOffPitchDeg: number;
  /** Peak toe bend during push-off (deg). */
  pushOffToeFlexDeg: number;
}

export const GAIT_PARAMS: Record<'Walk' | 'Run', GaitParams> = {
  Walk: {
    cycleDuration: 0.95,
    dutyFactor: 0.62,
    speed: 1.15,
    stanceStartZ: 0.36,
    stanceEndZ: -0.31735, // = 0.36 - 1.15 * 0.62 * 0.95 (exact treadmill distance ~0.68 m)
    stepWidthX: 0.11,
    swingClearanceY: 0.08,
    pelvisBaseOffsetY: -0.05,
    pelvisBobAmpY: 0.025,
    pelvisBobPolarity: 1,
    pelvisTwistDeg: 6,
    pelvisRollDeg: 3,
    torsoLeanDeg: 2,
    shoulderFlexRange: [-22, 26],
    elbowFlexRange: [12, 38],
    spineCounterTwistRatio: 0.7,
    heelStrikePitchDeg: 10,
    pushOffPitchDeg: -20,
    pushOffToeFlexDeg: 20,
  },
  Run: {
    cycleDuration: 0.606,
    dutyFactor: 0.35,
    speed: 3.4,
    stanceStartZ: 0.30,
    stanceEndZ: -0.42114, // = 0.30 - 3.4 * 0.35 * 0.606 (exact treadmill distance ~0.72 m)
    stepWidthX: 0.09,
    swingClearanceY: 0.30,
    pelvisBaseOffsetY: -0.08,
    pelvisBobAmpY: 0.05,
    pelvisBobPolarity: -1,
    pelvisTwistDeg: 9,
    pelvisRollDeg: 4,
    torsoLeanDeg: 12,
    shoulderFlexRange: [-45, 55],
    elbowFlexRange: [85, 105],
    spineCounterTwistRatio: 0.7,
    heelStrikePitchDeg: 8,
    pushOffPitchDeg: -20,
    pushOffToeFlexDeg: 20,
  },
};
