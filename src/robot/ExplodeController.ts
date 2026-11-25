import * as THREE from 'three';
import { easeInOutCubic, type ExplodeState } from '../core/AppState.ts';
import type { AssembledPart } from './RobotAssembler.ts';
import type { RobotRig } from './RobotRig.ts';

const MAX_FLOAT_HEIGHT = 0.12;
const GUIDE_LINE_THRESHOLD = 0.05;
/** Critically damped spring omega (~0.25s settling time: 4 / omega ≈ 0.25s => omega = 16) */
const SPRING_OMEGA = 16.0;

// Module-level scratch vectors (zero allocations in RAF)
const _currentCenterWorld = new THREE.Vector3();
const _anchorCenterWorld = new THREE.Vector3();

function stepCriticallyDamped(
  value: number,
  velocity: number,
  target: number,
  dt: number,
): [number, number] {
  const diff = value - target;
  const exp = Math.exp(-SPRING_OMEGA * dt);
  const newVal = target + (diff + (velocity + SPRING_OMEGA * diff) * dt) * exp;
  const newVel = (velocity - (velocity + SPRING_OMEGA * diff) * SPRING_OMEGA * dt) * exp;
  return [newVal, newVel];
}

export class ExplodeController {
  public readonly guideGroup = new THREE.Group();
  private readonly lineSegments: THREE.LineSegments;
  private readonly endpointDots: THREE.Points;
  private readonly linePosAttr: THREE.BufferAttribute;
  private readonly lineDistAttr: THREE.BufferAttribute;
  private readonly dotPosAttr: THREE.BufferAttribute;
  private readonly lineGeo: THREE.BufferGeometry;
  private readonly dotGeo: THREE.BufferGeometry;
  private readonly lineMat: THREE.LineDashedMaterial;
  private readonly dotMat: THREE.PointsMaterial;

  private readonly velocities = { t1: 0, t2: 0, t3: 0 };
  private readonly maxParts: number;

  constructor(private readonly parts: readonly AssembledPart[]) {
    this.maxParts = parts.length;
    this.guideGroup.name = 'assembly_guide_lines';

    // Pre-allocate LineSegments buffers (2 vertices per part)
    const vertexCount = this.maxParts * 2;
    const positions = new Float32Array(vertexCount * 3);
    const lineDistances = new Float32Array(vertexCount);

    this.linePosAttr = new THREE.BufferAttribute(positions, 3);
    this.linePosAttr.setUsage(THREE.DynamicDrawUsage);
    this.lineDistAttr = new THREE.BufferAttribute(lineDistances, 1);
    this.lineDistAttr.setUsage(THREE.DynamicDrawUsage);

    this.lineGeo = new THREE.BufferGeometry();
    this.lineGeo.setAttribute('position', this.linePosAttr);
    this.lineGeo.setAttribute('lineDistance', this.lineDistAttr);
    this.lineGeo.setDrawRange(0, 0);

    this.lineMat = new THREE.LineDashedMaterial({
      color: 0x38c6ff,
      dashSize: 0.016,
      gapSize: 0.010,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
    });

    this.lineSegments = new THREE.LineSegments(this.lineGeo, this.lineMat);
    this.lineSegments.frustumCulled = false;
    this.guideGroup.add(this.lineSegments);

    // Pre-allocate endpoint dots (2 endpoints per part)
    const dotPositions = new Float32Array(vertexCount * 3);
    this.dotPosAttr = new THREE.BufferAttribute(dotPositions, 3);
    this.dotPosAttr.setUsage(THREE.DynamicDrawUsage);

    this.dotGeo = new THREE.BufferGeometry();
    this.dotGeo.setAttribute('position', this.dotPosAttr);
    this.dotGeo.setDrawRange(0, 0);

    this.dotMat = new THREE.PointsMaterial({
      color: 0x62d9ff,
      size: 0.014,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });

    this.endpointDots = new THREE.Points(this.dotGeo, this.dotMat);
    this.endpointDots.frustumCulled = false;
    this.guideGroup.add(this.endpointDots);
  }

  /**
   * Step 1 (before AnimationEngine / IK): advances critically-damped springs
   * and sets whole-robot floatY so IK targets align exactly with the floated root.
   */
  public stepSpringAndFloat(
    dt: number,
    target: Readonly<ExplodeState>,
    current: ExplodeState,
    rig: RobotRig,
    snapImmediate = false,
  ): void {
    if (snapImmediate) {
      current.t1 = target.t1;
      current.t2 = target.t2;
      current.t3 = target.t3;
      this.velocities.t1 = 0;
      this.velocities.t2 = 0;
      this.velocities.t3 = 0;
    } else {
      const clampedDt = Math.min(dt, 0.05);
      [current.t1, this.velocities.t1] = stepCriticallyDamped(
        current.t1,
        this.velocities.t1,
        target.t1,
        clampedDt,
      );
      [current.t2, this.velocities.t2] = stepCriticallyDamped(
        current.t2,
        this.velocities.t2,
        target.t2,
        clampedDt,
      );
      [current.t3, this.velocities.t3] = stepCriticallyDamped(
        current.t3,
        this.velocities.t3,
        target.t3,
        clampedDt,
      );
    }

    const maxT = Math.max(current.t1, current.t2, current.t3);
    const floatY = MAX_FLOAT_HEIGHT * easeInOutCubic(maxT);
    rig.setRootFloatY(floatY);
  }

  /**
   * Step 2 (after AnimationEngine / IK): applies explode offsets to PartContainers
   * and updates the single-buffer 3D assembly guide lines. Zero heap allocations.
   */
  public applyOffsetsAndGuideLines(
    current: Readonly<ExplodeState>,
    rig: RobotRig,
    showGuideLines: boolean,
  ): void {
    const e1 = easeInOutCubic(current.t1);
    const e2 = easeInOutCubic(current.t2);
    const e3 = easeInOutCubic(current.t3);
    const maxT = Math.max(current.t1, current.t2, current.t3);

    for (let i = 0; i < this.parts.length; i++) {
      const part = this.parts[i]!;
      const factor = part.entry.tier === 1 ? e1 : part.entry.tier === 2 ? e2 : e3;
      part.container.position.set(
        part.basePos.x + part.explodeVector.x * factor,
        part.basePos.y + part.explodeVector.y * factor,
        part.basePos.z + part.explodeVector.z * factor,
      );
    }

    rig.updateWorldMatrices();

    const visible = showGuideLines && maxT > GUIDE_LINE_THRESHOLD;
    this.guideGroup.visible = visible;
    if (!visible) {
      this.lineGeo.setDrawRange(0, 0);
      this.dotGeo.setDrawRange(0, 0);
      return;
    }

    const posArr = this.linePosAttr.array as Float32Array;
    const distArr = this.lineDistAttr.array as Float32Array;
    const dotArr = this.dotPosAttr.array as Float32Array;
    let vertexIdx = 0;

    for (let i = 0; i < this.parts.length; i++) {
      const part = this.parts[i]!;
      if (!part.container.visible) continue;
      if (part.explodeVector.lengthSq() < 1e-8) continue;

      const factor = part.entry.tier === 1 ? e1 : part.entry.tier === 2 ? e2 : e3;
      if (factor <= 0.005) continue;

      _currentCenterWorld.copy(part.localCenter).applyMatrix4(part.container.matrixWorld);
      _anchorCenterWorld
        .copy(part.basePos)
        .add(part.localCenter)
        .applyMatrix4(part.parentJoint.matrixWorld);

      const dist = _currentCenterWorld.distanceTo(_anchorCenterWorld);
      if (dist < 0.004) continue;

      const base = vertexIdx * 3;
      posArr[base] = _anchorCenterWorld.x;
      posArr[base + 1] = _anchorCenterWorld.y;
      posArr[base + 2] = _anchorCenterWorld.z;
      posArr[base + 3] = _currentCenterWorld.x;
      posArr[base + 4] = _currentCenterWorld.y;
      posArr[base + 5] = _currentCenterWorld.z;

      dotArr[base] = _anchorCenterWorld.x;
      dotArr[base + 1] = _anchorCenterWorld.y;
      dotArr[base + 2] = _anchorCenterWorld.z;
      dotArr[base + 3] = _currentCenterWorld.x;
      dotArr[base + 4] = _currentCenterWorld.y;
      dotArr[base + 5] = _currentCenterWorld.z;

      distArr[vertexIdx] = 0;
      distArr[vertexIdx + 1] = dist;
      vertexIdx += 2;
    }

    this.lineGeo.setDrawRange(0, vertexIdx);
    this.dotGeo.setDrawRange(0, vertexIdx);
    this.linePosAttr.needsUpdate = true;
    this.lineDistAttr.needsUpdate = true;
    this.dotPosAttr.needsUpdate = true;
  }

  /**
   * Combined update (spring + float + offsets + guide lines) for deterministic tests/snaps.
   */
  public update(
    dt: number,
    target: Readonly<ExplodeState>,
    current: ExplodeState,
    rig: RobotRig,
    showGuideLines: boolean,
    snapImmediate = false,
  ): void {
    this.stepSpringAndFloat(dt, target, current, rig, snapImmediate);
    this.applyOffsetsAndGuideLines(current, rig, showGuideLines);
  }

  public dispose(): void {
    this.lineGeo.dispose();
    this.dotGeo.dispose();
    this.lineMat.dispose();
    this.dotMat.dispose();
  }
}
