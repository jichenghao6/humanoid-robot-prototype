import * as THREE from 'three';
import type { MaterialBank } from '../../config/materials.ts';
import type { PartCatalogEntry } from '../../config/partsCatalog.ts';
import {
  createConcentricDrumGeometry,
  createCurvedArmorPlate,
  createLoftedArmorShell,
} from '../GeometryHelpers.ts';
import type { PartRegistry } from '../PartRegistry.ts';
import {
  createBuckets,
  createCurvedTube,
  finalizeBuckets,
} from './PartBuilderUtils.ts';

// Head joint sits at world [0, 1.620, 0].
// Head proportions (Section 2.2): Y 1.605 ~ 1.800 (local Y -0.015 ~ +0.180), width 0.160 (w/ ear pods 0.200), depth 0.210.

function buildHeadHelmetShell(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Main aerodynamic cranial dome wrapping top, sides & rear (open in front for obsidian visor)
  const cranium = createLoftedArmorShell(
    [
      { y: 0.012, centerX: 0, centerZ: -0.014, rx: 0.056, rz: 0.068, squareness: 0.22, angleStart: -Math.PI * 1.12, angleEnd: Math.PI * 0.12 },
      { y: 0.055, centerX: 0, centerZ: -0.016, rx: 0.074, rz: 0.092, squareness: 0.26, angleStart: -Math.PI * 1.18, angleEnd: Math.PI * 0.18 },
      { y: 0.105, centerX: 0, centerZ: -0.012, rx: 0.080, rz: 0.102, squareness: 0.25, angleStart: -Math.PI * 1.24, angleEnd: Math.PI * 0.24 },
      { y: 0.148, centerX: 0, centerZ: -0.008, rx: 0.070, rz: 0.094, squareness: 0.22, angleStart: -Math.PI * 1.38, angleEnd: Math.PI * 0.38 },
      { y: 0.175, centerX: 0, centerZ: -0.010, rx: 0.042, rz: 0.064, squareness: 0.20, angleStart: -Math.PI * 1.50, angleEnd: Math.PI * 0.50 },
    ],
    22,
    0.012,
    false,
  );
  b.A.push(cranium);

  // Sculpted crown cap closing the top seamlessly up to Y = 0.180 (world 1.800m!)
  const crownCap = new THREE.SphereGeometry(0.068, 20, 12, 0, Math.PI * 2, 0, Math.PI * 0.45);
  crownCap.scale(0.95, 0.52, 1.28);
  crownCap.translate(0, 0.144, -0.010);
  b.A.push(crownCap);

  // Center cranial crest strip (cool-grey B aerospace fairing)
  const crest = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.016, -0.075),
      new THREE.Vector2(0.016, -0.075),
      new THREE.Vector2(0.013, 0.075),
      new THREE.Vector2(-0.013, 0.075),
    ],
    { thickness: 0.008, bevel: 0.003, bendRadiusY: 0.115, segments: 8 },
  );
  crest.rotateX(-Math.PI * 0.42);
  crest.translate(0, 0.172, -0.012);
  b.B.push(crest);

  // Sleek tapered jawline & pointed chin fairing (left & right cheek plates + chin)
  for (const sx of [-1, 1] as const) {
    const cheek = createCurvedArmorPlate(
      [
        new THREE.Vector2(-0.022, -0.042),
        new THREE.Vector2(0.032, -0.030),
        new THREE.Vector2(0.036, 0.040),
        new THREE.Vector2(-0.018, 0.046),
      ],
      { thickness: 0.010, bevel: 0.003, bendRadiusX: 0.095, segments: 6 },
    );
    cheek.rotateY(sx * Math.PI * 0.36);
    cheek.translate(sx * 0.058, 0.042, 0.044);
    b.A.push(cheek);
  }

  // Pointed chin cap
  const chin = createCurvedArmorPlate(
    [
      new THREE.Vector2(0, -0.024),
      new THREE.Vector2(-0.030, 0.012),
      new THREE.Vector2(0, 0.020),
      new THREE.Vector2(0.030, 0.012),
    ],
    { thickness: 0.012, bevel: 0.004, bendRadiusX: 0.07, segments: 6 },
  );
  chin.rotateX(0.28);
  chin.translate(0, 0.004, 0.068);
  b.A.push(chin);

  // Dark rear occipital vent recess (BACK view detail)
  const rearVent = new THREE.BoxGeometry(0.062, 0.066, 0.024);
  rearVent.rotateX(0.18);
  rearVent.translate(0, 0.060, -0.092);
  b.C.push(rearVent);

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

function buildHeadVisor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Curved obsidian-black optical face visor (Grade D, strong clearcoat reflection)
  const visorShell = createLoftedArmorShell(
    [
      { y: 0.004, centerX: 0, centerZ: 0.012, rx: 0.038, rz: 0.062, squareness: 0.20, angleStart: Math.PI * 0.22, angleEnd: Math.PI * 0.78 },
      { y: 0.048, centerX: 0, centerZ: 0.014, rx: 0.064, rz: 0.086, squareness: 0.28, angleStart: Math.PI * 0.16, angleEnd: Math.PI * 0.84 },
      { y: 0.105, centerX: 0, centerZ: 0.012, rx: 0.068, rz: 0.092, squareness: 0.30, angleStart: Math.PI * 0.16, angleEnd: Math.PI * 0.84 },
      { y: 0.148, centerX: 0, centerZ: 0.006, rx: 0.056, rz: 0.082, squareness: 0.26, angleStart: Math.PI * 0.20, angleEnd: Math.PI * 0.80 },
    ],
    18,
    0.008,
    false,
  );
  b.D.push(visorShell);

  // Iconic downward-tapering V-shaped pair of electric-blue eye LED strips (FRONT & SIDE anchor!)
  for (const sx of [-1, 1] as const) {
    const eyeCurve = createCurvedTube(
      [
        new THREE.Vector3(sx * 0.048, 0.118, 0.082),
        new THREE.Vector3(sx * 0.044, 0.082, 0.093),
        new THREE.Vector3(sx * 0.025, 0.034, 0.086),
        new THREE.Vector3(sx * 0.013, 0.016, 0.078),
      ],
      0.0042,
      18,
      8,
    );
    b.E.push(eyeCurve);
  }

  // Subtle horizontal brow sensor slit inside visor (matches reference forehead cue)
  const browLed = createCurvedTube(
    [
      new THREE.Vector3(-0.026, 0.132, 0.082),
      new THREE.Vector3(0.0, 0.134, 0.088),
      new THREE.Vector3(0.026, 0.132, 0.082),
    ],
    0.0028,
    12,
    6,
  );
  b.E.push(browLed);

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

function buildHeadEarPod(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Canonical L temple sensor pod @ x = +0.084 (total width w/ pods = 0.200 m)
  const drum = createConcentricDrumGeometry({
    outerRadius: 0.032,
    width: 0.028,
    innerHubRadius: 0.016,
    boltCount: 6,
    segments: 24,
  });
  drum.translate(0.083, 0.082, -0.008);
  b.C.push(drum);

  // Outer cool-grey sensor rim
  const rim = new THREE.TorusGeometry(0.030, 0.0032, 8, 24);
  rim.rotateY(Math.PI / 2);
  rim.translate(0.097, 0.082, -0.008);
  b.B.push(rim);

  // Complete circular glowing electric-blue ear LED ring (key SIDE & FRONT view anchor!)
  const ledRing = new THREE.TorusGeometry(0.022, 0.0032, 10, 28);
  ledRing.rotateY(Math.PI / 2);
  ledRing.translate(0.099, 0.082, -0.008);
  b.E.push(ledRing);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

function buildHeadAiCore(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Internal Tier 3 edge AI inference core with stacked copper/aluminum heatsink fins
  const coreBlock = new THREE.BoxGeometry(0.072, 0.068, 0.076);
  coreBlock.translate(0, 0.086, -0.008);
  b.F.push(coreBlock);

  for (let i = 0; i < 6; i++) {
    const fin = new THREE.BoxGeometry(0.082, 0.004, 0.068);
    fin.translate(0, 0.064 + i * 0.009, -0.008);
    b.copper.push(fin);
  }

  const opticalBus = new THREE.CylinderGeometry(0.014, 0.014, 0.042, 12);
  opticalBus.rotateX(Math.PI / 2);
  opticalBus.translate(0, 0.086, 0.032);
  b.chrome.push(opticalBus);

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

export function registerHeadBuilders(registry: PartRegistry): void {
  registry.register('head_helmet_shell', buildHeadHelmetShell);
  registry.register('head_visor', buildHeadVisor);
  registry.register('head_ear_pod', buildHeadEarPod);
  registry.register('head_ai_core', buildHeadAiCore);
}
