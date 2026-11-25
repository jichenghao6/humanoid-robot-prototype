import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { GAIT_PARAMS } from '../src/config/gait.ts';
import { createMaterialBank } from '../src/config/materials.ts';
import { PARTS_CATALOG, computeCatalogStats } from '../src/config/partsCatalog.ts';
import { RIG_JOINT_MAP, computeRigDofStats } from '../src/config/rig.ts';
import { AnimationEngine } from '../src/robot/AnimationEngine.ts';
import { ExplodeController } from '../src/robot/ExplodeController.ts';
import { createCurvedArmorPlate, mirrorGeometryX } from '../src/robot/GeometryHelpers.ts';
import { PartRegistry } from '../src/robot/PartRegistry.ts';
import { RenderModeController } from '../src/robot/RenderModeController.ts';
import { PART_RAYCAST_LAYER, RobotAssembler } from '../src/robot/RobotAssembler.ts';
import { RobotRig } from '../src/robot/RobotRig.ts';
import { registerArmBuilders } from '../src/robot/parts/ArmBuilder.ts';
import { registerHeadBuilders } from '../src/robot/parts/HeadBuilder.ts';
import { registerLegBuilders } from '../src/robot/parts/LegBuilder.ts';
import { registerTorsoBuilders } from '../src/robot/parts/TorsoBuilder.ts';

describe('7.4.1 Catalog & Rig consistency', () => {
  it('has unique partIds, valid parent joints, complete _L/_R pairs, valid tiers, 57 parts and 35 active DOFs', () => {
    const seenIds = new Set<string>();
    const leftBases = new Set<string>();
    const rightBases = new Set<string>();

    for (const entry of PARTS_CATALOG) {
      expect(seenIds.has(entry.partId)).toBe(false);
      seenIds.add(entry.partId);

      expect(RIG_JOINT_MAP.has(entry.parentJoint)).toBe(true);
      expect([1, 2, 3]).toContain(entry.tier);

      if (entry.partId.endsWith('_L')) {
        leftBases.add(entry.baseId);
        expect(entry.side).toBe('L');
      } else if (entry.partId.endsWith('_R')) {
        rightBases.add(entry.baseId);
        expect(entry.side).toBe('R');
      } else {
        expect(entry.side).toBeNull();
      }
    }

    expect(leftBases.size).toBe(rightBases.size);
    for (const base of leftBases) {
      expect(rightBases.has(base)).toBe(true);
    }

    const catalogStats = computeCatalogStats();
    expect(catalogStats.totalParts).toBe(57);
    expect(catalogStats.entryTypes).toBe(35);
    expect(catalogStats.totalMassKg).toBeGreaterThan(79);
    expect(catalogStats.totalMassKg).toBeLessThan(83);

    const dofStats = computeRigDofStats();
    expect(dofStats.torsoActiveDof).toBe(9);
    expect(dofStats.armActiveDof).toBe(12);
    expect(dofStats.legActiveDof).toBe(14);
    expect(dofStats.totalActiveDof).toBe(35);
    expect(dofStats.fingerDof).toBe(10);
    expect(dofStats.floatingBaseDof).toBe(6);
  });
});

describe('7.4.2 Mirror symmetry & axis signs', () => {
  it('points both hands toward +Z when shoulder_L and shoulder_R flex = 90°', () => {
    const rig = new RobotRig();
    rig.setSemanticAngles('shoulder_L', 90, 0, 0);
    rig.setSemanticAngles('shoulder_R', 90, 0, 0);
    rig.updateWorldMatrices();

    const sL = rig.getJointWorldPosition('shoulder_L', new THREE.Vector3());
    const wL = rig.getJointWorldPosition('wrist_L', new THREE.Vector3());
    const sR = rig.getJointWorldPosition('shoulder_R', new THREE.Vector3());
    const wR = rig.getJointWorldPosition('wrist_R', new THREE.Vector3());

    const dirL = wL.clone().sub(sL).normalize();
    const dirR = wR.clone().sub(sR).normalize();

    expect(dirL.z).toBeGreaterThan(0.95);
    expect(dirR.z).toBeGreaterThan(0.95);
    expect(dirL.x).toBeCloseTo(-dirR.x, 4);
    expect(dirL.y).toBeCloseTo(dirR.y, 4);
    expect(dirL.z).toBeCloseTo(dirR.z, 4);
  });

  it('points left hand toward +X and right hand toward -X when shoulder abd = 90°', () => {
    const rig = new RobotRig();
    rig.setSemanticAngles('shoulder_L', 0, 90, 0);
    rig.setSemanticAngles('shoulder_R', 0, 90, 0);
    rig.updateWorldMatrices();

    const sL = rig.getJointWorldPosition('shoulder_L', new THREE.Vector3());
    const wL = rig.getJointWorldPosition('wrist_L', new THREE.Vector3());
    const sR = rig.getJointWorldPosition('shoulder_R', new THREE.Vector3());
    const wR = rig.getJointWorldPosition('wrist_R', new THREE.Vector3());

    const dirL = wL.clone().sub(sL).normalize();
    const dirR = wR.clone().sub(sR).normalize();

    expect(dirL.x).toBeGreaterThan(0.95);
    expect(dirR.x).toBeLessThan(-0.95);
    expect(dirL.x).toBeCloseTo(-dirR.x, 4);
  });

  it('places ankle behind knee (-Z) when knee.flex = 90°', () => {
    const rig = new RobotRig();
    rig.setSemanticAngles('knee_L', 90, 0, 0);
    rig.setSemanticAngles('knee_R', 90, 0, 0);
    rig.updateWorldMatrices();

    const kL = rig.getJointWorldPosition('knee_L', new THREE.Vector3());
    const aL = rig.getJointWorldPosition('ankle_L', new THREE.Vector3());
    const kR = rig.getJointWorldPosition('knee_R', new THREE.Vector3());
    const aR = rig.getJointWorldPosition('ankle_R', new THREE.Vector3());

    expect(aL.z - kL.z).toBeLessThan(-0.45);
    expect(aR.z - kR.z).toBeLessThan(-0.45);
  });

  it('curls left fingers inward toward -X palm and right fingers toward +X palm with 3-phalanx linkage', () => {
    const rig = new RobotRig();
    rig.updateWorldMatrices();

    const baseL = rig.getJointWorldPosition('finger_index_L', new THREE.Vector3());
    const baseR = rig.getJointWorldPosition('finger_index_R', new THREE.Vector3());
    const openTipL = rig.getFingerTipWorldPosition('index', 'L', new THREE.Vector3());
    expect(openTipL.y).toBeLessThan(baseL.y - 0.06);

    rig.setHandCurl('L', 90);
    rig.setHandCurl('R', 90);
    rig.updateWorldMatrices();

    const curledTipL = rig.getFingerTipWorldPosition('index', 'L', new THREE.Vector3());
    const curledTipR = rig.getFingerTipWorldPosition('index', 'R', new THREE.Vector3());
    // Left palm faces -X, right palm faces +X; curled fingertip must fold inward without sideways Z drift
    expect(curledTipL.x - baseL.x).toBeLessThan(-0.015);
    expect(curledTipR.x - baseR.x).toBeGreaterThan(0.015);
    expect(Math.abs(curledTipL.z - baseL.z)).toBeLessThan(0.002);
    expect(curledTipL.x - baseL.x).toBeCloseTo(-(curledTipR.x - baseR.x), 4);
  });
});

describe('7.4.3 Joint limit clamping', () => {
  it('clamps out-of-range semantic angles to configured limits', () => {
    const rig = new RobotRig();

    // Knee cannot hyperextend (< 0) or exceed 135
    expect(rig.setSemanticAngles('knee_L', -30).flex).toBe(0);
    expect(rig.setSemanticAngles('knee_L', 200).flex).toBe(135);

    // Shoulder limits: flex -50..160, abd -8..120, twist -60..60
    const sHigh = rig.setSemanticAngles('shoulder_R', 220, 180, 95);
    expect(sHigh.flex).toBe(160);
    expect(sHigh.abd).toBe(120);
    expect(sHigh.twist).toBe(60);

    const sLow = rig.setSemanticAngles('shoulder_L', -120, -40, -90);
    expect(sLow.flex).toBe(-50);
    expect(sLow.abd).toBe(-8);
    expect(sLow.twist).toBe(-60);

    // Elbow has only flex 0..135; abd/twist must stay 0
    const eRes = rig.setSemanticAngles('elbow_L', -10, 45, 45);
    expect(eRes.flex).toBe(0);
    expect(eRes.abd).toBe(0);
    expect(eRes.twist).toBe(0);
  });
});

describe('7.4.4 Gait ground contact, zero-slip velocity & IK limits', () => {
  for (const mode of ['Walk', 'Run'] as const) {
    it(`${mode}: stance foot stays on ground (<2mm), moves at -v (<3% err), and IK angles stay in limits`, () => {
      const rig = new RobotRig();
      const engine = new AnimationEngine();
      const params = GAIT_PARAMS[mode];
      const samples = 150; // >= 120 frames per cycle
      const dt = params.cycleDuration / samples;

      const prevContactZ: Record<'L' | 'R', number | null> = { L: null, R: null };
      const prevInStance: Record<'L' | 'R', boolean> = { L: false, R: false };
      const scratch = new THREE.Vector3();
      const heelPos = new THREE.Vector3();
      const ballPos = new THREE.Vector3();
      const centerPos = new THREE.Vector3();

      let maxGroundErr = 0;
      let maxVelRelErr = 0;
      let stanceVelChecks = 0;

      for (let i = 0; i < samples; i++) {
        const phase = i / samples;
        const { leftSample, rightSample } = engine.evaluateGaitAtPhase(mode, phase, rig);
        rig.updateWorldMatrices();

        const groundBaseY = rig.getRootFloatY();

        // Check all leg joint limits
        for (const side of ['L', 'R'] as const) {
          for (const prefix of ['hip', 'knee', 'ankle', 'toe'] as const) {
            const jId = `${prefix}_${side}`;
            const def = rig.getJointDef(jId);
            const ang = rig.getSemanticAngles(jId);
            if (def.limits.flex) {
              expect(ang.flex).toBeGreaterThanOrEqual(def.limits.flex.min - 1e-6);
              expect(ang.flex).toBeLessThanOrEqual(def.limits.flex.max + 1e-6);
            }
            if (def.limits.abd) {
              expect(ang.abd).toBeGreaterThanOrEqual(def.limits.abd.min - 1e-6);
              expect(ang.abd).toBeLessThanOrEqual(def.limits.abd.max + 1e-6);
            }
            if (def.limits.twist) {
              expect(ang.twist).toBeGreaterThanOrEqual(def.limits.twist.min - 1e-6);
              expect(ang.twist).toBeLessThanOrEqual(def.limits.twist.max + 1e-6);
            }
          }

          const sample = side === 'L' ? leftSample : rightSample;
          const refs = rig.soleRefs[side];
          refs.heel.getWorldPosition(heelPos);
          refs.ball.getWorldPosition(ballPos);
          refs.center.getWorldPosition(centerPos);

          if (sample.inStance) {
            const minSoleY = rig.getLowestSoleY(side, scratch);
            const groundErr = Math.abs(minSoleY - groundBaseY);
            if (groundErr > maxGroundErr) maxGroundErr = groundErr;
            expect(groundErr).toBeLessThan(0.002); // < 2 mm

            // Active ground contact point Z (heel during dorsiflexion, ball during push-off, center during flat)
            const contactZ =
              sample.footPitchDeg > 0.05
                ? heelPos.z
                : sample.footPitchDeg < -0.05
                  ? ballPos.z
                  : centerPos.z;

            // Check Z velocity against -v during continuous flat/contact stance sub-segments
            const sameContactSubPhase =
              prevInStance[side] &&
              prevContactZ[side] !== null &&
              Math.abs(sample.footPitchDeg) < 1e-4;

            if (sameContactSubPhase) {
              const vz = (contactZ - prevContactZ[side]!) / dt;
              const relErr = Math.abs(vz - -params.speed) / params.speed;
              if (relErr > maxVelRelErr) maxVelRelErr = relErr;
              expect(relErr).toBeLessThan(0.03); // < 3%
              stanceVelChecks += 1;
            }

            prevContactZ[side] = Math.abs(sample.footPitchDeg) < 1e-4 ? contactZ : null;
            prevInStance[side] = true;
          } else {
            prevInStance[side] = false;
            prevContactZ[side] = null;
          }
        }
      }

      expect(stanceVelChecks).toBeGreaterThan(20);
      expect(maxGroundErr).toBeLessThan(0.002);
      expect(maxVelRelErr).toBeLessThan(0.03);
    });
  }
});

describe('7.4.5 Full explode during Walk never penetrates ground', () => {
  it('keeps every part world bounding box min.y >= -0.005 m across full Walk cycle at {1, 1, 1}', () => {
    const rig = new RobotRig();
    const materials = createMaterialBank();
    const registry = new PartRegistry();
    registerHeadBuilders(registry);
    registerTorsoBuilders(registry);
    registerArmBuilders(registry, ['L', 'R']);
    registerLegBuilders(registry, ['L', 'R']);
    const assembler = new RobotAssembler();
    assembler.assemble(rig, registry, materials);

    // Verify assembled bind-pose total height is 1.800 m (±0.03 m)
    const fullBox = new THREE.Box3().setFromObject(rig.root);
    expect(fullBox.min.y).toBeGreaterThanOrEqual(-0.005);
    expect(fullBox.max.y).toBeGreaterThan(1.77);
    expect(fullBox.max.y).toBeLessThan(1.83);

    // Ensure no Object3D in the entire assembled robot uses negative scale (Section 3.1.4)
    rig.root.traverse((obj) => {
      expect(obj.scale.x).toBeGreaterThan(0);
      expect(obj.scale.y).toBeGreaterThan(0);
      expect(obj.scale.z).toBeGreaterThan(0);
    });

    const explode = new ExplodeController(assembler.partsList);
    const engine = new AnimationEngine();

    const target = { t1: 1, t2: 1, t3: 1 };
    const current = { t1: 1, t2: 1, t3: 1 };
    const box = new THREE.Box3();
    const samples = 60;
    let lowestY = Infinity;
    let lowestPartId = '';

    for (let i = 0; i < samples; i++) {
      const phase = i / samples;
      // Apply explode float before IK so IK targets align with floated root
      explode.update(0.016, target, current, rig, false, true);
      engine.evaluateGaitAtPhase('Walk', phase, rig);
      explode.update(0.016, target, current, rig, false, true);

      for (const part of assembler.partsList) {
        box.setFromObject(part.container);
        if (!box.isEmpty() && box.min.y < lowestY) {
          lowestY = box.min.y;
          lowestPartId = part.entry.partId;
        }
      }
    }

    materials.dispose();
    explode.dispose();

    expect(lowestY, `lowest part was ${lowestPartId} at y=${lowestY}`).toBeGreaterThanOrEqual(
      -0.005,
    );
  });

  it('mirrorGeometryX mirrors X coordinates and preserves positive outward winding', () => {
    const box = new THREE.BoxGeometry(0.2, 0.3, 0.4);
    box.translate(0.5, 0.0, 0.0);
    const mirrored = mirrorGeometryX(box);
    mirrored.computeBoundingBox();

    expect(mirrored.boundingBox!.min.x).toBeCloseTo(-0.6, 5);
    expect(mirrored.boundingBox!.max.x).toBeCloseTo(-0.4, 5);

    // Verify signed volume of mirrored closed mesh is positive (outward normals!)
    const pos = mirrored.getAttribute('position');
    const idx = mirrored.index;
    const v0 = new THREE.Vector3();
    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();
    const cross = new THREE.Vector3();
    let volume6 = 0;
    const triCount = idx ? idx.count / 3 : pos.count / 3;
    for (let t = 0; t < triCount; t++) {
      const a = idx ? idx.getX(t * 3) : t * 3;
      const b = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
      const c = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
      v0.fromBufferAttribute(pos, a);
      v1.fromBufferAttribute(pos, b);
      v2.fromBufferAttribute(pos, c);
      cross.crossVectors(v1, v2);
      volume6 += v0.dot(cross);
    }
    expect(volume6 / 6).toBeCloseTo(0.2 * 0.3 * 0.4, 5);

    box.dispose();
    mirrored.dispose();
  });

  it('Jump & Wave procedural motions respect ground pinning, airborne arc, and pose targets', () => {
    const rig = new RobotRig();
    const engine = new AnimationEngine();
    const scratch = new THREE.Vector3();

    // Sample Jump across full 2.4s cycle
    engine.setMode('Jump', rig, true);
    let maxSoleY = -Infinity;
    let crouchGroundErr = 0;
    for (let i = 0; i < 48; i++) {
      engine.update(0.05, 1.0, true, rig);
      rig.updateWorldMatrices();
      const t = (i + 1) * 0.05;
      const soleY = rig.getLowestSoleY('L', scratch);
      if (soleY > maxSoleY) maxSoleY = soleY;
      // During deep crouch (t ~ 0.55s) and landing cushion (t ~ 1.5s), feet must stay pinned to ground
      if ((t >= 0.2 && t <= 0.6) || (t >= 1.4 && t <= 2.3)) {
        crouchGroundErr = Math.max(crouchGroundErr, Math.abs(soleY));
      }
    }
    expect(maxSoleY).toBeGreaterThan(0.18); // Airborne flight clearly leaves ground
    expect(crouchGroundErr).toBeLessThan(0.005); // IK pins feet during crouch/landing

    // Sample Wave at peak wave phase (t = 1.0s) and verify continuous loop past t = 3.0s
    engine.setMode('Wave', rig, true);
    engine.update(1.0, 1.0, true, rig);
    expect(rig.getSemanticAngles('shoulder_R').abd).toBeCloseTo(85, 0);
    expect(rig.getSemanticAngles('elbow_R').flex).toBeCloseTo(100, 0);
    expect(rig.getSemanticAngles('head').abd).toBeCloseTo(8, 0);
    expect(rig.getSemanticAngles('shoulder_L').flex).toBeCloseTo(0, 0);

    engine.update(1.99, 1.0, true, rig); // t = 2.99s
    const abdPre = rig.getSemanticAngles('shoulder_R').abd;
    const twistPre = rig.getSemanticAngles('shoulder_R').twist;
    engine.update(0.02, 1.0, true, rig); // t = 3.01s
    expect(Math.abs(rig.getSemanticAngles('shoulder_R').abd - abdPre)).toBeLessThan(1.0);
    expect(Math.abs(rig.getSemanticAngles('shoulder_R').twist - twistPre)).toBeLessThan(5.0);

    // Verify Jump takeoff (0.80s) and landing (1.25s) transitions are continuous
    for (const boundary of [0.80, 1.25]) {
      engine.setMode('Jump', rig, true);
      engine.update(boundary - 0.001, 1.0, true, rig);
      const kneePre = rig.getSemanticAngles('knee_L').flex;
      const hipPre = rig.getSemanticAngles('hip_L').flex;
      engine.update(0.002, 1.0, true, rig);
      expect(Math.abs(rig.getSemanticAngles('knee_L').flex - kneePre)).toBeLessThan(1.5);
      expect(Math.abs(rig.getSemanticAngles('hip_L').flex - hipPre)).toBeLessThan(1.5);
    }
  });

  it('keeps cranium top normals outward, mesh count <= 174 (draw call budget), and Run/Jump full explode >= -0.005 m', () => {
    const rig = new RobotRig();
    const materials = createMaterialBank();
    const registry = new PartRegistry();
    registerHeadBuilders(registry);
    registerTorsoBuilders(registry);
    registerArmBuilders(registry, ['L', 'R']);
    registerLegBuilders(registry, ['L', 'R']);
    const assembler = new RobotAssembler();
    assembler.assemble(rig, registry, materials);

    // Instanced fingers reduce total robot meshes from 200 -> 174 (so High mode w/ Reflector <= 350 draw calls)
    expect(assembler.allPartMeshes.length).toBeLessThanOrEqual(174);

    // Verify no twisted/inverted outer dome normals on head_helmet_shell top cranium (y in 0.149..0.174)
    const headMesh = assembler.getPart('head_helmet_shell')!.meshes[0]!;
    const pos = headMesh.geometry.getAttribute('position');
    const nrm = headMesh.geometry.getAttribute('normal');
    let outerTopCount = 0;
    let outerDownwardCount = 0;
    for (let i = 0; i < pos.count; i += 3) {
      const cy = (pos.getY(i) + pos.getY(i + 1) + pos.getY(i + 2)) / 3;
      if (cy > 0.149 && cy < 0.174) {
        const cx = (pos.getX(i) + pos.getX(i + 1) + pos.getX(i + 2)) / 3;
        const cz = (pos.getZ(i) + pos.getZ(i + 1) + pos.getZ(i + 2)) / 3 + 0.009;
        const nx = (nrm.getX(i) + nrm.getX(i + 1) + nrm.getX(i + 2)) / 3;
        const ny = (nrm.getY(i) + nrm.getY(i + 1) + nrm.getY(i + 2)) / 3;
        const nz = (nrm.getZ(i) + nrm.getZ(i + 1) + nrm.getZ(i + 2)) / 3;
        const radialDot = cx * nx + (cy - 0.11) * ny + cz * nz;
        if (radialDot > 0) {
          outerTopCount++;
          if (ny < -0.05) outerDownwardCount++;
        }
      }
    }
    expect(outerTopCount).toBeGreaterThanOrEqual(40);
    expect(outerDownwardCount).toBe(0);

    // Verify full explode {1,1,1} never penetrates floor during Run or Jump
    const explode = new ExplodeController(assembler.partsList);
    const engine = new AnimationEngine();
    const target = { t1: 1, t2: 1, t3: 1 };
    const current = { t1: 1, t2: 1, t3: 1 };
    const box = new THREE.Box3();

    for (const mode of ['Run', 'Jump'] as const) {
      engine.setMode(mode, rig, true);
      let lowestY = Infinity;
      const steps = 80;
      const dt = 2.4 / steps;
      for (let i = 0; i < steps; i++) {
        explode.stepSpringAndFloat(0.016, target, current, rig, true);
        if (mode === 'Run') engine.evaluateGaitAtPhase('Run', i / steps, rig);
        else engine.update(dt, 1.0, true, rig);
        explode.applyOffsetsAndGuideLines(current, rig, false);
        for (const p of assembler.partsList) {
          box.setFromObject(p.container);
          if (!box.isEmpty() && box.min.y < lowestY) lowestY = box.min.y;
        }
      }
      expect(lowestY, `${mode} min.y=${lowestY}`).toBeGreaterThanOrEqual(-0.005);
    }

    materials.dispose();
    explode.dispose();
  });

  it('produces smooth non-flat creased normals on curved armor and preserves them on mirrorGeometryX', () => {
    const plate = createCurvedArmorPlate(
      [
        new THREE.Vector2(-0.08, -0.08),
        new THREE.Vector2(0.08, -0.08),
        new THREE.Vector2(0.08, 0.08),
        new THREE.Vector2(-0.08, 0.08),
      ],
      { thickness: 0.016, bevel: 0.004, bendRadiusX: 0.14, segments: 8, crownZ: 0.015 },
    );
    const nrm = plate.getAttribute('normal');
    // Verify no NaN normals and adjacent vertices of a single curved triangle have distinct smooth normals
    let hasSmoothGradient = false;
    for (let i = 0; i < nrm.count; i += 3) {
      expect(Number.isFinite(nrm.getX(i))).toBe(true);
      expect(Number.isFinite(nrm.getY(i))).toBe(true);
      expect(Number.isFinite(nrm.getZ(i))).toBe(true);
      const dx = Math.abs(nrm.getX(i) - nrm.getX(i + 1));
      if (dx > 1e-3) hasSmoothGradient = true;
    }
    expect(hasSmoothGradient).toBe(true);

    // Mirror X must preserve smooth normals (negating X, not flattening per-triangle)
    const mirrored = mirrorGeometryX(plate);
    const mNrm = mirrored.getAttribute('normal');
    let mirroredSmooth = false;
    for (let i = 0; i < mNrm.count; i += 3) {
      if (Math.abs(mNrm.getX(i) - mNrm.getX(i + 1)) > 1e-3) mirroredSmooth = true;
    }
    expect(mirroredSmooth).toBe(true);
    plate.dispose();
    mirrored.dispose();
  });

  it('X-Ray turns Tier 1 and Tier 2 A/B outer armor translucent while keeping C/F internal structures PBR and E LEDs glowing', () => {
    const rig = new RobotRig();
    const materials = createMaterialBank();
    const registry = new PartRegistry();
    registerHeadBuilders(registry);
    registerTorsoBuilders(registry);
    registerArmBuilders(registry, ['L', 'R']);
    registerLegBuilders(registry, ['L', 'R']);
    const assembler = new RobotAssembler();
    assembler.assemble(rig, registry, materials);

    const renderModeCtrl = new RenderModeController(assembler.partsList, materials);
    renderModeCtrl.setMode('X-Ray');

    const footPart = assembler.getPart('foot_assembly_L')!;
    const footWhiteUpper = footPart.meshes.find((m) => m.userData.materialGrade === 'A')!;
    const footDarkSole = footPart.meshes.find((m) => m.userData.materialGrade === 'C')!;
    const footLed = footPart.meshes.find((m) => m.userData.isLed)!;

    expect(footWhiteUpper.material).toBe(materials.xrayShell);
    expect(footDarkSole.material).toBe(materials.grades.C);
    expect(footLed.material).toBe(materials.grades.E);

    const handPart = assembler.getPart('hand_assembly_L')!;
    const handWhiteDorsal = handPart.meshes.find((m) => m.userData.materialGrade === 'A')!;
    const handDarkPhalanges = handPart.meshes.find((m) => m.userData.materialGrade === 'C')!;
    expect(handWhiteDorsal.material).toBe(materials.xrayShell);
    expect(handDarkPhalanges.material).toBe(materials.grades.C);

    materials.dispose();
  });

  it('PartRegistry.useGlbOverrides replaces procedural part visual cleanly while preserving metadata and layers', () => {
    const rig = new RobotRig();
    const materials = createMaterialBank();
    const registry = new PartRegistry();
    registerHeadBuilders(registry);
    registerTorsoBuilders(registry);
    registerArmBuilders(registry, ['L', 'R']);
    registerLegBuilders(registry, ['L', 'R']);
    const assembler = new RobotAssembler();
    assembler.assemble(rig, registry, materials);

    const customGroup = new THREE.Group();
    const customGeo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    const customMat = new THREE.MeshStandardMaterial({ color: 0xff0000 });
    const customMesh = new THREE.Mesh(customGeo, customMat);
    customGroup.add(customMesh);

    registry.registerGlbOverride('head_ai_core', customGroup);
    expect(registry.hasGlbOverride('head_ai_core')).toBe(true);
    registry.useGlbOverrides = true;

    assembler.applyPartOverrides(registry, materials, ['head_ai_core']);
    const part = assembler.getPart('head_ai_core')!;
    expect(part.meshes.length).toBe(1);
    expect(part.meshes[0]!.userData.partId).toBe('head_ai_core');
    expect(part.meshes[0]!.userData.tier).toBe(3);
    expect(part.meshes[0]!.layers.isEnabled(PART_RAYCAST_LAYER)).toBe(true);
    expect(assembler.allPartMeshes.includes(part.meshes[0]!)).toBe(true);

    customGeo.dispose();
    customMat.dispose();
    materials.dispose();
  });

  it('bridges pelvis_gimbal_base and lumbar_actuator across spine_lower with zero vertical gap across joint limits', () => {
    const rig = new RobotRig();
    const materials = createMaterialBank();
    const registry = new PartRegistry();
    registerHeadBuilders(registry);
    registerTorsoBuilders(registry);
    registerArmBuilders(registry, ['L', 'R']);
    registerLegBuilders(registry, ['L', 'R']);
    const assembler = new RobotAssembler();
    assembler.assemble(rig, registry, materials);

    const pelvisBase = assembler.getPart('pelvis_gimbal_base')!;
    const pelvisFront = assembler.getPart('pelvis_front_armor')!;
    const lumbar = assembler.getPart('lumbar_actuator')!;
    const abdomen = assembler.getPart('abdomen_hex_plate')!;
    const chestChassis = assembler.getPart('torso_main_chassis')!;

    const pelvisBox = new THREE.Box3();
    const pelvisFrontBox = new THREE.Box3();
    const lumbarBox = new THREE.Box3();
    const abdomenBox = new THREE.Box3();
    const chestBox = new THREE.Box3();

    const poses: ReadonlyArray<[number, number, number]> = [
      [0, 0, 0],
      [25, 0, 0],
      [-15, 0, 0],
      [0, 15, 0],
      [0, 0, 25],
    ];

    for (const [flex, abd, twist] of poses) {
      rig.resetToBindPose();
      rig.setSemanticAngles('spine_lower', flex, abd, twist);
      rig.updateWorldMatrices();

      pelvisBox.setFromObject(pelvisBase.container);
      pelvisFrontBox.setFromObject(pelvisFront.container);
      lumbarBox.setFromObject(lumbar.container);
      abdomenBox.setFromObject(abdomen.container);
      chestBox.setFromObject(chestChassis.container);

      expect(pelvisBox.max.y).toBeGreaterThan(lumbarBox.min.y + 0.03);
      expect(lumbarBox.max.y).toBeGreaterThanOrEqual(chestBox.min.y - 0.002);
      if (flex === 0 && abd === 0 && twist === 0) {
        expect(pelvisFrontBox.max.y).toBeGreaterThanOrEqual(abdomenBox.min.y);
      }
    }

    materials.dispose();
  });
});
