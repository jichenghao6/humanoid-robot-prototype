import * as THREE from 'three';
import type { MaterialBank } from '../config/materials.ts';
import {
  PARTS_CATALOG,
  type PartCatalogEntry,
  type PartTier,
} from '../config/partsCatalog.ts';
import { RIG_JOINTS, type Side } from '../config/rig.ts';
import type { HandFingerMeshes, PartRegistry } from './PartRegistry.ts';
import type { RobotRig } from './RobotRig.ts';

export const PART_RAYCAST_LAYER = 1;
export const BLOOM_LAYER = 2;

const _m1 = new THREE.Matrix4();
const _m2 = new THREE.Matrix4();
const _m3 = new THREE.Matrix4();
const _scaleMat = new THREE.Matrix4();
const _outMat = new THREE.Matrix4();
const _RADIAL_RATIOS: readonly [number, number, number] = [1.0, 0.88, 0.76];

export interface AssembledPart {
  entry: PartCatalogEntry;
  parentJoint: THREE.Group;
  container: THREE.Group;
  visualRoot: THREE.Object3D;
  basePos: THREE.Vector3;
  explodeVector: THREE.Vector3;
  /** Local bounding center of the part mesh inside container (used for guide lines & focus). */
  localCenter: THREE.Vector3;
  meshes: THREE.Mesh[];
  ledMeshes: THREE.Mesh[];
}

export class RobotAssembler {
  public readonly parts = new Map<string, AssembledPart>();
  public readonly partsList: AssembledPart[] = [];
  public readonly allPartMeshes: THREE.Mesh[] = [];

  public assemble(
    rig: RobotRig,
    registry: PartRegistry,
    materials: MaterialBank,
  ): void {
    const tmpBox = new THREE.Box3();
    const tmpCenter = new THREE.Vector3();

    // 1. Create PartContainer for every BOM part under its parent joint
    for (const entry of PARTS_CATALOG) {
      const parentJoint = rig.getPivot(entry.parentJoint);

      const container = new THREE.Group();
      container.name = `part_container_${entry.partId}`;
      container.userData.partId = entry.partId;
      container.userData.tier = entry.tier;
      parentJoint.add(container);

      const visual = registry.build(entry, materials);
      container.add(visual);

      const meshes: THREE.Mesh[] = [];
      const ledMeshes: THREE.Mesh[] = [];

      visual.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.userData.partId = entry.partId;
          obj.userData.tier = entry.tier;
          obj.layers.enable(PART_RAYCAST_LAYER);
          if (obj.userData.isLed) {
            obj.layers.enable(BLOOM_LAYER);
            ledMeshes.push(obj);
          }
          meshes.push(obj);
          this.allPartMeshes.push(obj);
        }
      });

      // Compute local bounding center inside container
      container.updateMatrixWorld(true);
      tmpBox.setFromObject(visual);
      if (!tmpBox.isEmpty()) {
        tmpBox.getCenter(tmpCenter);
        container.worldToLocal(tmpCenter);
      } else {
        tmpCenter.set(0, 0, 0);
      }

      const assembled: AssembledPart = {
        entry,
        parentJoint,
        container,
        visualRoot: visual,
        basePos: container.position.clone(),
        explodeVector: new THREE.Vector3(
          entry.explodeVector[0],
          entry.explodeVector[1],
          entry.explodeVector[2],
        ),
        localCenter: tmpCenter.clone(),
        meshes,
        ledMeshes,
      };

      this.parts.set(entry.partId, assembled);
      this.partsList.push(assembled);
    }

    // 2. Mount assembly-exception sub-joints (fingers -> hand_assembly, toe -> foot_assembly)
    for (const jointDef of RIG_JOINTS) {
      if (!jointDef.mountPartId) continue;
      const targetPart = this.parts.get(jointDef.mountPartId);
      if (!targetPart) {
        throw new Error(
          `Assembly-exception mountPartId '${jointDef.mountPartId}' not found for joint '${jointDef.id}'`,
        );
      }
      rig.mountJointOnContainer(jointDef.id, targetPart.container);
    }

    // 3. Attach detailed or placeholder toe cap visual onto toe_L/R
    for (const side of ['L', 'R'] as const) {
      const toePivot = rig.getPivot(`toe_${side}`);
      const footPart = this.parts.get(`foot_assembly_${side}`);
      const customToe = registry.buildToe(side, materials);

      if (customToe) {
        toePivot.add(customToe);
        customToe.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            obj.userData.partId = `foot_assembly_${side}`;
            obj.userData.tier = 2 satisfies PartTier;
            obj.layers.enable(PART_RAYCAST_LAYER);
            if (obj.userData.isLed) {
              obj.layers.enable(BLOOM_LAYER);
              footPart?.ledMeshes.push(obj);
            }
            footPart?.meshes.push(obj);
            this.allPartMeshes.push(obj);
          }
        });
      } else {
        const toeGeo = new THREE.BoxGeometry(0.112, 0.034, 0.082);
        const toeMesh = new THREE.Mesh(toeGeo, materials.tierPlaceholder[2]);
        toeMesh.position.set(0, -0.017, 0.041);
        toeMesh.castShadow = true;
        toeMesh.receiveShadow = true;
        toeMesh.userData.partId = `foot_assembly_${side}`;
        toeMesh.userData.tier = 2 satisfies PartTier;
        toeMesh.layers.enable(PART_RAYCAST_LAYER);
        toePivot.add(toeMesh);

        if (footPart) {
          footPart.meshes.push(toeMesh);
          this.allPartMeshes.push(toeMesh);
        }
      }
    }

    // 4. Attach instanced articulated 3-phalanx fingers onto hand_assembly containers
    const handFingerMeshes: Partial<Record<Side, HandFingerMeshes>> = {};
    for (const side of ['L', 'R'] as const) {
      const handPart = this.parts.get(`hand_assembly_${side}`);
      const fingerMeshes = registry.buildHandFingers(side, materials);
      if (!handPart || !fingerMeshes) continue;

      handFingerMeshes[side] = fingerMeshes;
      for (const im of [fingerMeshes.phalangesC, fingerMeshes.padsA]) {
        im.userData.partId = `hand_assembly_${side}`;
        im.userData.tier = 2 satisfies PartTier;
        im.layers.enable(PART_RAYCAST_LAYER);
        handPart.container.add(im);
        handPart.meshes.push(im);
        this.allPartMeshes.push(im);
      }
    }

    rig.setFingerSyncHandler((side: Side) => {
      const meshes = handFingerMeshes[side];
      if (!meshes) return;
      const fingers = rig.fingerNodes[side];
      for (let f = 0; f < fingers.length; f++) {
        const node = fingers[f]!;
        node.phalanx1.updateMatrix();
        node.phalanx2.updateMatrix();
        node.phalanx3.updateMatrix();

        _m1.copy(node.phalanx1.matrix);
        _m2.multiplyMatrices(_m1, node.phalanx2.matrix);
        _m3.multiplyMatrices(_m2, node.phalanx3.matrix);

        const r0 = node.baseRadius * _RADIAL_RATIOS[0];
        const r1 = node.baseRadius * _RADIAL_RATIOS[1];
        const r2 = node.baseRadius * _RADIAL_RATIOS[2];

        _scaleMat.makeScale(r0, node.segLengths[0], r0);
        _outMat.multiplyMatrices(_m1, _scaleMat);
        meshes.phalangesC.setMatrixAt(f * 3 + 0, _outMat);
        meshes.padsA.setMatrixAt(f * 2 + 0, _outMat);

        _scaleMat.makeScale(r1, node.segLengths[1], r1);
        _outMat.multiplyMatrices(_m2, _scaleMat);
        meshes.phalangesC.setMatrixAt(f * 3 + 1, _outMat);
        meshes.padsA.setMatrixAt(f * 2 + 1, _outMat);

        _scaleMat.makeScale(r2, node.segLengths[2], r2);
        _outMat.multiplyMatrices(_m3, _scaleMat);
        meshes.phalangesC.setMatrixAt(f * 3 + 2, _outMat);
      }

      meshes.phalangesC.instanceMatrix.needsUpdate = true;
      meshes.padsA.instanceMatrix.needsUpdate = true;
      meshes.phalangesC.computeBoundingBox();
      meshes.phalangesC.computeBoundingSphere();
      meshes.padsA.computeBoundingBox();
      meshes.padsA.computeBoundingSphere();
    });

    rig.updateWorldMatrices();
  }

  /**
   * Replaces a part's primary visual subtree (used by GLB override loader) while
   * keeping joint hierarchy, sub-pivots (toe/fingers), explode vectors, and inspector intact.
   */
  public replacePartVisual(partId: string, nextVisual: THREE.Object3D): void {
    const part = this.parts.get(partId);
    if (!part) return;

    const oldVisual = part.visualRoot;
    const removedMeshes = new Set<THREE.Mesh>();
    oldVisual.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        removedMeshes.add(obj);
        obj.geometry.dispose();
      }
    });

    part.container.remove(oldVisual);
    part.meshes = part.meshes.filter((m) => !removedMeshes.has(m));
    part.ledMeshes = part.ledMeshes.filter((m) => !removedMeshes.has(m));
    for (let i = this.allPartMeshes.length - 1; i >= 0; i--) {
      if (removedMeshes.has(this.allPartMeshes[i]!)) {
        this.allPartMeshes.splice(i, 1);
      }
    }

    part.container.add(nextVisual);
    part.visualRoot = nextVisual;

    nextVisual.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        obj.userData.partId = part.entry.partId;
        obj.userData.tier = part.entry.tier;
        obj.layers.enable(PART_RAYCAST_LAYER);
        if (obj.userData.isLed) {
          obj.layers.enable(BLOOM_LAYER);
          part.ledMeshes.push(obj);
        }
        part.meshes.push(obj);
        this.allPartMeshes.push(obj);
      }
    });

    part.container.updateMatrixWorld(true);
    const tmpBox = new THREE.Box3().setFromObject(nextVisual);
    if (!tmpBox.isEmpty()) {
      tmpBox.getCenter(part.localCenter);
      part.container.worldToLocal(part.localCenter);
    } else {
      part.localCenter.set(0, 0, 0);
    }
  }

  public applyPartOverrides(
    registry: PartRegistry,
    materials: MaterialBank,
    partIds?: readonly string[],
  ): void {
    const ids = partIds ?? this.partsList.map((p) => p.entry.partId);
    for (const partId of ids) {
      const part = this.parts.get(partId);
      if (!part) continue;
      const nextVisual = registry.build(part.entry, materials);
      this.replacePartVisual(partId, nextVisual);
    }
  }

  public getPart(partId: string): AssembledPart | undefined {
    return this.parts.get(partId);
  }

  public getPartTier(partId: string): PartTier | undefined {
    return this.parts.get(partId)?.entry.tier;
  }
}
