import * as THREE from 'three';
import type { MaterialBank } from '../../config/materials.ts';
import type { Side } from '../../config/rig.ts';
import { mergeGeometriesSafe, mirrorGeometryX } from '../GeometryHelpers.ts';

export interface MaterialBuckets {
  A: THREE.BufferGeometry[];
  B: THREE.BufferGeometry[];
  C: THREE.BufferGeometry[];
  D: THREE.BufferGeometry[];
  E: THREE.BufferGeometry[];
  F: THREE.BufferGeometry[];
  copper: THREE.BufferGeometry[];
  chrome: THREE.BufferGeometry[];
}

export function createBuckets(): MaterialBuckets {
  return { A: [], B: [], C: [], D: [], E: [], F: [], copper: [], chrome: [] };
}

/**
 * Merges each material bucket into one draw call per material,
 * optionally mirroring geometry across X via mirrorGeometryX (never negative Object3D scale).
 */
export function finalizeBuckets(
  partId: string,
  tier: number,
  side: Side | null,
  buckets: MaterialBuckets,
  materials: MaterialBank,
): THREE.Group {
  const group = new THREE.Group();
  group.name = `part_mesh_${partId}`;

  const emit = (
    geos: THREE.BufferGeometry[],
    mat: THREE.Material,
    gradeKey: string,
    isLed = false,
  ): void => {
    if (geos.length === 0) return;
    let merged = mergeGeometriesSafe(geos, true);
    if (side === 'R') {
      const mirrored = mirrorGeometryX(merged);
      merged.dispose();
      merged = mirrored;
    }
    const mesh = new THREE.Mesh(merged, mat);
    mesh.castShadow = !isLed;
    mesh.receiveShadow = !isLed;
    mesh.userData.partId = partId;
    mesh.userData.tier = tier;
    mesh.userData.materialGrade = gradeKey;
    if (isLed) {
      mesh.userData.isLed = true;
    }
    group.add(mesh);
  };

  emit(buckets.A, materials.grades.A, 'A');
  emit(buckets.B, materials.grades.B, 'B');
  emit(buckets.C, materials.grades.C, 'C');
  emit(buckets.D, materials.grades.D, 'D');
  emit(buckets.F, materials.grades.F, 'F');
  emit(buckets.copper, materials.gradeFCopper, 'copper');
  emit(buckets.chrome, materials.gradeFChrome, 'chrome');
  emit(buckets.E, materials.grades.E, 'E', true);

  return group;
}

/**
 * Adds a dark Grade C recessed bezel + slightly proud (+1.5 mm) Grade E LED strip
 * along a 3D line segment oriented with outward surface normal.
 */
export function addRecessedLedStrip(
  buckets: MaterialBuckets,
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  outNormal: THREE.Vector3,
  ledWidth: number,
  bezelWidth: number,
): void {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  const len = dir.length();
  const yAxis = dir.clone().normalize();
  const zAxis = outNormal.clone().normalize();
  const xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();
  zAxis.crossVectors(xAxis, yAxis).normalize();

  const basis = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
  const mid = p0.clone().add(p1).multiplyScalar(0.5);

  const bezel = new THREE.BoxGeometry(bezelWidth, len + 0.008, 0.007);
  bezel.applyMatrix4(basis);
  bezel.translate(mid.x, mid.y, mid.z);
  buckets.C.push(bezel);

  const led = new THREE.BoxGeometry(ledWidth, len, 0.004);
  led.applyMatrix4(basis);
  const ledPos = mid.clone().addScaledVector(zAxis, 0.0032);
  led.translate(ledPos.x, ledPos.y, ledPos.z);
  buckets.E.push(led);
}

/**
 * Creates a smooth curved TubeGeometry along a CatmullRomCurve3 (for cables, heatpipes, curved LEDs).
 */
export function createCurvedTube(
  points: readonly THREE.Vector3[],
  radius: number,
  tubularSegments = 20,
  radialSegments = 8,
): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3([...points]);
  return new THREE.TubeGeometry(curve, tubularSegments, radius, radialSegments, false);
}
