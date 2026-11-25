export interface SegmentProportion {
  yMin: number;
  yMax: number;
  widthX: number;
  depthZ: number;
}

export const ROBOT_TOTAL_HEIGHT = 1.800;

export const PROPORTIONS: Record<
  | 'head'
  | 'neck'
  | 'chest'
  | 'abdomen'
  | 'pelvis'
  | 'upperArm'
  | 'forearm'
  | 'hand'
  | 'thigh'
  | 'shin'
  | 'foot',
  SegmentProportion
> = {
  head: { yMin: 1.605, yMax: 1.800, widthX: 0.160, depthZ: 0.210 },
  neck: { yMin: 1.545, yMax: 1.605, widthX: 0.090, depthZ: 0.100 },
  chest: { yMin: 1.370, yMax: 1.560, widthX: 0.400, depthZ: 0.270 },
  abdomen: { yMin: 1.175, yMax: 1.370, widthX: 0.220, depthZ: 0.180 },
  pelvis: { yMin: 0.960, yMax: 1.175, widthX: 0.290, depthZ: 0.210 },
  upperArm: { yMin: 1.250, yMax: 1.480, widthX: 0.100, depthZ: 0.110 },
  forearm: { yMin: 0.970, yMax: 1.250, widthX: 0.090, depthZ: 0.095 },
  hand: { yMin: 0.800, yMax: 0.970, widthX: 0.085, depthZ: 0.035 },
  thigh: { yMin: 0.660, yMax: 1.040, widthX: 0.180, depthZ: 0.190 },
  shin: { yMin: 0.140, yMax: 0.660, widthX: 0.140, depthZ: 0.160 },
  foot: { yMin: 0.000, yMax: 0.140, widthX: 0.130, depthZ: 0.310 },
};

export const FOOT_GEOMETRY_METRICS = {
  ankleHeight: 0.140,
  heelOffsetZ: -0.070,
  ballOffsetZ: 0.140,
  toeTipOffsetFromToeZ: 0.080,
  toeHeight: 0.035,
  soleCenterOffsetZ: 0.020,
} as const;
