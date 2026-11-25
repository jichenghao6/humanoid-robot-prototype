import * as THREE from 'three';
import type { MaterialBank } from '../../config/materials.ts';
import type { PartCatalogEntry } from '../../config/partsCatalog.ts';
import type { Side } from '../../config/rig.ts';
import {
  createConcentricDrumGeometry,
  createCurvedArmorPlate,
  createHydraulicStrut,
  createLoftedArmorShell,
  createStrutSegment,
} from '../GeometryHelpers.ts';
import type { PartRegistry } from '../PartRegistry.ts';
import {
  addRecessedLedStrip,
  createBuckets,
  finalizeBuckets,
} from './PartBuilderUtils.ts';

// ---------------------------------------------------------------------------
// 1. Hip Actuator (parent: pelvis; canonical L @ [+0.125, -0.010, 0])
// ---------------------------------------------------------------------------
function buildHipActuator(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const drum = createConcentricDrumGeometry({
    outerRadius: 0.058,
    width: 0.082,
    innerHubRadius: 0.030,
    boltCount: 8,
    segments: 28,
  });
  drum.translate(0.125, -0.010, 0.0);
  b.C.push(drum);

  // Inner pelvic mounting yoke collar
  const yoke = new THREE.CylinderGeometry(0.044, 0.048, 0.045, 20);
  yoke.rotateZ(Math.PI / 2);
  yoke.translate(0.085, -0.010, 0.0);
  b.C.push(yoke);

  // Machined outer cool-grey trim ring
  const trimRing = new THREE.TorusGeometry(0.048, 0.0045, 10, 28);
  trimRing.rotateY(Math.PI / 2);
  trimRing.translate(0.168, -0.010, 0.0);
  b.B.push(trimRing);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 2. Thigh Front Armor (parent: hip; knee @ [+0.040, -0.380, +0.010])
// ---------------------------------------------------------------------------
function buildThighFrontArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Main sculpted pearl-white front-outer thigh shell (angle 0 = +X outer, PI/2 = +Z front)
  const mainShell = createLoftedArmorShell(
    [
      { y: -0.340, centerX: 0.036, centerZ: 0.012, rx: 0.064, rz: 0.072, squareness: 0.35, angleStart: -0.22, angleEnd: Math.PI * 0.88 },
      { y: -0.250, centerX: 0.028, centerZ: 0.012, rx: 0.076, rz: 0.084, squareness: 0.35, angleStart: -0.28, angleEnd: Math.PI * 0.90 },
      { y: -0.140, centerX: 0.020, centerZ: 0.010, rx: 0.086, rz: 0.092, squareness: 0.32, angleStart: -0.30, angleEnd: Math.PI * 0.90 },
      { y: -0.035, centerX: 0.014, centerZ: 0.006, rx: 0.088, rz: 0.090, squareness: 0.28, angleStart: -0.25, angleEnd: Math.PI * 0.85 },
      { y: 0.032, centerX: 0.018, centerZ: 0.002, rx: 0.082, rz: 0.074, squareness: 0.30, angleStart: -0.18, angleEnd: Math.PI * 0.55 },
    ],
    20,
    0.014,
    false,
  );
  b.A.push(mainShell);

  // High outer hip armor crest plate (iconic upward silhouette from reference image)
  const hipFlare = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.045, -0.065),
      new THREE.Vector2(0.050, -0.055),
      new THREE.Vector2(0.040, 0.055),
      new THREE.Vector2(-0.015, 0.075),
      new THREE.Vector2(-0.052, 0.025),
    ],
    { thickness: 0.014, bevel: 0.004, bendRadiusX: 0.14, bendRadiusY: 0.28, segments: 8, crownZ: 0.008 },
  );
  hipFlare.rotateY(Math.PI * 0.42);
  hipFlare.translate(0.084, -0.015, 0.024);
  b.A.push(hipFlare);

  // Cool-grey upper-inner secondary fairing & lower knee-collar bevel
  const innerTrim = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.038, -0.110),
      new THREE.Vector2(0.032, -0.095),
      new THREE.Vector2(0.036, 0.090),
      new THREE.Vector2(-0.030, 0.105),
    ],
    { thickness: 0.010, bevel: 0.003, bendRadiusX: 0.10, segments: 6 },
  );
  innerTrim.rotateY(-Math.PI * 0.28);
  innerTrim.translate(-0.034, -0.165, 0.058);
  b.B.push(innerTrim);

  const lowerCollar = createLoftedArmorShell(
    [
      { y: -0.355, centerX: 0.037, centerZ: 0.014, rx: 0.062, rz: 0.069, squareness: 0.35, angleStart: -0.15, angleEnd: Math.PI * 0.80 },
      { y: -0.325, centerX: 0.035, centerZ: 0.014, rx: 0.066, rz: 0.073, squareness: 0.35, angleStart: -0.15, angleEnd: Math.PI * 0.80 },
    ],
    14,
    0.008,
    false,
  );
  b.B.push(lowerCollar);

  // Signature front-outer long diagonal recessed electric-blue LED light slot (clearly visible on surface in FRONT & SIDE)
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.088, -0.052, 0.072),
    new THREE.Vector3(0.086, -0.315, 0.058),
    new THREE.Vector3(0.74, 0.0, 0.67),
    0.009,
    0.022,
  );

  // Secondary outer-side accent LED aligned with side-view flow line
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.106, -0.085, 0.014),
    new THREE.Vector3(0.100, -0.265, 0.016),
    new THREE.Vector3(0.98, 0.0, 0.16),
    0.0065,
    0.016,
  );

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 3. Thigh Rear Armor (parent: hip; deep dark radiator vent + vertical LED)
// ---------------------------------------------------------------------------
function buildThighRearArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Outer-rear white wing
  const outerWing = createLoftedArmorShell(
    [
      { y: -0.325, centerX: 0.035, centerZ: 0.008, rx: 0.064, rz: 0.072, squareness: 0.30, angleStart: -Math.PI * 0.40, angleEnd: -0.16 },
      { y: -0.190, centerX: 0.025, centerZ: 0.006, rx: 0.082, rz: 0.088, squareness: 0.30, angleStart: -Math.PI * 0.42, angleEnd: -0.18 },
      { y: -0.055, centerX: 0.016, centerZ: 0.002, rx: 0.084, rz: 0.086, squareness: 0.28, angleStart: -Math.PI * 0.40, angleEnd: -0.16 },
    ],
    10,
    0.012,
    false,
  );
  b.A.push(outerWing);

  // Inner-rear white wing
  const innerWing = createLoftedArmorShell(
    [
      { y: -0.325, centerX: 0.035, centerZ: 0.008, rx: 0.064, rz: 0.072, squareness: 0.30, angleStart: -Math.PI * 1.05, angleEnd: -Math.PI * 0.62 },
      { y: -0.190, centerX: 0.025, centerZ: 0.006, rx: 0.082, rz: 0.088, squareness: 0.30, angleStart: -Math.PI * 1.05, angleEnd: -Math.PI * 0.60 },
      { y: -0.055, centerX: 0.016, centerZ: 0.002, rx: 0.084, rz: 0.086, squareness: 0.28, angleStart: -Math.PI * 1.02, angleEnd: -Math.PI * 0.62 },
    ],
    10,
    0.012,
    false,
  );
  b.A.push(innerWing);

  // Deep central dark titanium heatsink radiator bay + louvers
  const bayBack = new THREE.BoxGeometry(0.058, 0.245, 0.028);
  bayBack.translate(0.024, -0.190, -0.066);
  b.C.push(bayBack);

  for (let i = 0; i < 6; i++) {
    const fin = new THREE.BoxGeometry(0.052, 0.007, 0.016);
    fin.rotateX(0.32);
    fin.translate(0.024, -0.095 - i * 0.034, -0.078);
    b.C.push(fin);
  }

  // Rear vertical electric-blue LED bar (BACK view anchor)
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.021, -0.095, -0.083),
    new THREE.Vector3(0.031, -0.280, -0.078),
    new THREE.Vector3(0, 0, -1),
    0.009,
    0.020,
  );

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 4. Femur Frame (Tier 3 internal skeleton + twin hydraulic pistons)
// ---------------------------------------------------------------------------
function buildFemurFrame(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Main titanium structural femur spar
  b.C.push(
    createStrutSegment(
      new THREE.Vector3(0.004, -0.020, 0.000),
      new THREE.Vector3(0.038, -0.360, 0.008),
      0.024,
      0.028,
      14,
    ),
  );

  // Hip upper yoke block & lower knee clevis forks
  const upperYoke = new THREE.BoxGeometry(0.068, 0.054, 0.066);
  upperYoke.translate(0.006, -0.028, 0.000);
  b.C.push(upperYoke);

  const midRib = new THREE.BoxGeometry(0.056, 0.028, 0.088);
  midRib.translate(0.022, -0.185, 0.004);
  b.F.push(midRib);

  const lowerClevisL = new THREE.BoxGeometry(0.018, 0.076, 0.062);
  lowerClevisL.translate(0.070, -0.352, 0.008);
  const lowerClevisR = new THREE.BoxGeometry(0.018, 0.076, 0.062);
  lowerClevisR.translate(0.008, -0.352, 0.008);
  b.C.push(lowerClevisL, lowerClevisR);

  // Twin linear actuator pistons (front + rear)
  const frontPiston = createHydraulicStrut(
    new THREE.Vector3(0.014, -0.055, 0.044),
    new THREE.Vector3(0.036, -0.330, 0.038),
    0.013,
    0.007,
  );
  b.F.push(frontPiston.barrel);
  b.chrome.push(frontPiston.rod);

  const rearPiston = createHydraulicStrut(
    new THREE.Vector3(0.016, -0.055, -0.042),
    new THREE.Vector3(0.038, -0.330, -0.032),
    0.012,
    0.007,
  );
  b.C.push(rearPiston.barrel);
  b.copper.push(rearPiston.rod);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 5. Knee Patella Guard (Tier 1, parent: patella; floating angular kneecap)
// ---------------------------------------------------------------------------
function buildKneePatellaGuard(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Main diamond-shield curved kneecap
  const shield = createCurvedArmorPlate(
    [
      new THREE.Vector2(0.0, 0.066),
      new THREE.Vector2(-0.044, 0.024),
      new THREE.Vector2(-0.038, -0.038),
      new THREE.Vector2(0.0, -0.064),
      new THREE.Vector2(0.038, -0.038),
      new THREE.Vector2(0.044, 0.024),
    ],
    {
      thickness: 0.016,
      bevel: 0.005,
      bendRadiusX: 0.085,
      bendRadiusY: 0.16,
      segments: 8,
      crownZ: 0.012,
    },
  );
  shield.translate(0.002, 0.002, 0.076);
  b.A.push(shield);

  // Cool-grey upper chamfer cap & dark inner mounting bracket
  const upperCap = createCurvedArmorPlate(
    [
      new THREE.Vector2(0.0, 0.056),
      new THREE.Vector2(-0.032, 0.030),
      new THREE.Vector2(0.0, 0.014),
      new THREE.Vector2(0.032, 0.030),
    ],
    { thickness: 0.009, bevel: 0.003, bendRadiusX: 0.085, segments: 6 },
  );
  upperCap.translate(0.002, 0.016, 0.082);
  b.B.push(upperCap);

  const bracket = new THREE.BoxGeometry(0.044, 0.042, 0.034);
  bracket.translate(0.002, 0.000, 0.056);
  b.C.push(bracket);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 6. Knee Drum Actuator (Tier 2, parent: knee; signature side-view drum!)
// ---------------------------------------------------------------------------
function buildKneeDrumActuator(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const drum = createConcentricDrumGeometry({
    outerRadius: 0.056,
    width: 0.114,
    innerHubRadius: 0.028,
    boltCount: 8,
    segments: 32,
  });
  b.C.push(drum);

  // Machined concentric cool-grey bearing ring visible on both outer & inner sides
  for (const sx of [-1, 1] as const) {
    const outerRing = new THREE.TorusGeometry(0.045, 0.0042, 10, 32);
    outerRing.rotateY(Math.PI / 2);
    outerRing.translate(sx * 0.058, 0, 0);
    b.B.push(outerRing);

    const innerCap = new THREE.CylinderGeometry(0.018, 0.020, 0.008, 20);
    innerCap.rotateZ(Math.PI / 2);
    innerCap.translate(sx * 0.060, 0, 0);
    b.F.push(innerCap);
  }

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 7. Shin Front Armor (Tier 1, parent: knee; ankle @ [+0.040, -0.520, -0.030])
// ---------------------------------------------------------------------------
function buildShinFrontArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Sleek aerodynamic blade-tapered pearl-white shin guard
  const shinShell = createLoftedArmorShell(
    [
      { y: -0.460, centerX: 0.036, centerZ: -0.022, rx: 0.046, rz: 0.048, squareness: 0.38, angleStart: -0.20, angleEnd: Math.PI * 0.88 },
      { y: -0.340, centerX: 0.028, centerZ: -0.012, rx: 0.056, rz: 0.062, squareness: 0.38, angleStart: -0.25, angleEnd: Math.PI * 0.92 },
      { y: -0.200, centerX: 0.018, centerZ: 0.000, rx: 0.068, rz: 0.076, squareness: 0.36, angleStart: -0.28, angleEnd: Math.PI * 0.92 },
      { y: -0.090, centerX: 0.010, centerZ: 0.008, rx: 0.066, rz: 0.074, squareness: 0.34, angleStart: -0.22, angleEnd: Math.PI * 0.88 },
      { y: -0.048, centerX: 0.006, centerZ: 0.012, rx: 0.056, rz: 0.062, squareness: 0.32, angleStart: -0.10, angleEnd: Math.PI * 0.80 },
    ],
    20,
    0.013,
    false,
  );
  b.A.push(shinShell);

  // Sharp sculpted front shin ridge plate (adds heroic athletic line)
  const shinCrest = createCurvedArmorPlate(
    [
      new THREE.Vector2(0.0, 0.175),
      new THREE.Vector2(-0.028, 0.120),
      new THREE.Vector2(-0.022, -0.165),
      new THREE.Vector2(0.0, -0.195),
      new THREE.Vector2(0.022, -0.165),
      new THREE.Vector2(0.028, 0.120),
    ],
    { thickness: 0.010, bevel: 0.004, bendRadiusX: 0.075, segments: 8, crownZ: 0.008 },
  );
  shinCrest.rotateX(0.065);
  shinCrest.rotateZ(0.075);
  shinCrest.translate(0.022, -0.255, 0.066);
  b.A.push(shinCrest);

  // Cool-grey lower ankle cuff transition guard
  const cuff = createLoftedArmorShell(
    [
      { y: -0.482, centerX: 0.038, centerZ: -0.024, rx: 0.045, rz: 0.047, squareness: 0.35, angleStart: -0.15, angleEnd: Math.PI * 0.85 },
      { y: -0.445, centerX: 0.035, centerZ: -0.020, rx: 0.048, rz: 0.050, squareness: 0.35, angleStart: -0.15, angleEnd: Math.PI * 0.85 },
    ],
    14,
    0.009,
    false,
  );
  b.B.push(cuff);

  // Outer-side segmented electric-blue LED strips (2 segments proud on outer shin ridge, FRONT & SIDE anchor)
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.077, -0.115, 0.044),
    new THREE.Vector3(0.082, -0.265, 0.028),
    new THREE.Vector3(0.84, 0.0, 0.54),
    0.008,
    0.019,
  );
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.082, -0.292, 0.022),
    new THREE.Vector3(0.081, -0.432, 0.004),
    new THREE.Vector3(0.86, 0.0, 0.51),
    0.007,
    0.017,
  );

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 8. Calf Rear Armor (Tier 1, parent: knee; split calf wings + Achilles LED)
// ---------------------------------------------------------------------------
function buildCalfRearArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Outer-rear sculpted white calf wing (bulges athletically at upper calf, tapers sharply to ankle)
  const outerCalf = createLoftedArmorShell(
    [
      { y: -0.440, centerX: 0.034, centerZ: -0.024, rx: 0.042, rz: 0.048, squareness: 0.28, angleStart: -Math.PI * 0.42, angleEnd: -0.15 },
      { y: -0.320, centerX: 0.026, centerZ: -0.022, rx: 0.056, rz: 0.068, squareness: 0.28, angleStart: -Math.PI * 0.44, angleEnd: -0.16 },
      { y: -0.185, centerX: 0.018, centerZ: -0.020, rx: 0.070, rz: 0.088, squareness: 0.30, angleStart: -Math.PI * 0.44, angleEnd: -0.16 },
      { y: -0.085, centerX: 0.010, centerZ: -0.008, rx: 0.062, rz: 0.070, squareness: 0.28, angleStart: -Math.PI * 0.40, angleEnd: -0.15 },
    ],
    10,
    0.012,
    false,
  );
  b.A.push(outerCalf);

  // Inner-rear sculpted white calf wing
  const innerCalf = createLoftedArmorShell(
    [
      { y: -0.440, centerX: 0.034, centerZ: -0.024, rx: 0.042, rz: 0.048, squareness: 0.28, angleStart: -Math.PI * 1.05, angleEnd: -Math.PI * 0.58 },
      { y: -0.320, centerX: 0.026, centerZ: -0.022, rx: 0.056, rz: 0.068, squareness: 0.28, angleStart: -Math.PI * 1.06, angleEnd: -Math.PI * 0.56 },
      { y: -0.185, centerX: 0.018, centerZ: -0.020, rx: 0.070, rz: 0.088, squareness: 0.30, angleStart: -Math.PI * 1.06, angleEnd: -Math.PI * 0.56 },
      { y: -0.085, centerX: 0.010, centerZ: -0.008, rx: 0.062, rz: 0.070, squareness: 0.28, angleStart: -Math.PI * 1.02, angleEnd: -Math.PI * 0.60 },
    ],
    10,
    0.012,
    false,
  );
  b.A.push(innerCalf);

  // Deep dark titanium Achilles channel &散热 ribs
  const achillesBay = new THREE.BoxGeometry(0.040, 0.320, 0.026);
  achillesBay.rotateX(-0.07);
  achillesBay.translate(0.024, -0.265, -0.066);
  b.C.push(achillesBay);

  for (let i = 0; i < 5; i++) {
    const rib = new THREE.BoxGeometry(0.036, 0.006, 0.016);
    rib.translate(0.022 + i * 0.003, -0.145 - i * 0.042, -0.084 + i * 0.003);
    b.C.push(rib);
  }

  // Vertical electric-blue Achilles LED bar (BACK view anchor)
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.021, -0.155, -0.094),
    new THREE.Vector3(0.033, -0.395, -0.074),
    new THREE.Vector3(0, 0, -1),
    0.008,
    0.018,
  );

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 9. Tibia Frame (Tier 3, parent: knee; structural twin-spar + ankle pushrods)
// ---------------------------------------------------------------------------
function buildTibiaFrame(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Main front structural titanium tibia beam
  b.C.push(
    createStrutSegment(
      new THREE.Vector3(0.004, -0.045, 0.012),
      new THREE.Vector3(0.038, -0.495, -0.020),
      0.020,
      0.024,
      14,
    ),
  );

  // Upper & lower structural collars
  const upperCollar = new THREE.BoxGeometry(0.062, 0.042, 0.062);
  upperCollar.translate(0.006, -0.062, 0.002);
  const midBrace = new THREE.BoxGeometry(0.052, 0.024, 0.068);
  midBrace.translate(0.022, -0.260, -0.010);
  const ankleFork = new THREE.BoxGeometry(0.054, 0.048, 0.052);
  ankleFork.translate(0.038, -0.478, -0.026);
  b.F.push(upperCollar, midBrace);
  b.C.push(ankleFork);

  // Dual rear ankle hydraulic pushrods (outer & inner)
  const pushOuter = createHydraulicStrut(
    new THREE.Vector3(0.032, -0.095, -0.030),
    new THREE.Vector3(0.056, -0.485, -0.046),
    0.010,
    0.006,
  );
  b.C.push(pushOuter.barrel);
  b.chrome.push(pushOuter.rod);

  const pushInner = createHydraulicStrut(
    new THREE.Vector3(-0.012, -0.095, -0.030),
    new THREE.Vector3(0.020, -0.485, -0.046),
    0.010,
    0.006,
  );
  b.C.push(pushInner.barrel);
  b.copper.push(pushInner.rod);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 10. Ankle Actuator (Tier 2, parent: ankle; exposed drum + signature blue LED ring!)
// ---------------------------------------------------------------------------
function buildAnkleActuator(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const drum = createConcentricDrumGeometry({
    outerRadius: 0.040,
    width: 0.090,
    innerHubRadius: 0.020,
    boltCount: 6,
    segments: 28,
  });
  b.C.push(drum);

  // 2-axis gimbal stirrup connecting down into foot chassis
  const stirrup = new THREE.BoxGeometry(0.054, 0.058, 0.054);
  stirrup.translate(0, -0.036, 0.006);
  b.C.push(stirrup);

  // Recessed circular groove + glowing electric-blue LED ring on outer & inner faces (FRONT/SIDE anchor!)
  for (const sx of [-1, 1] as const) {
    const bezel = new THREE.TorusGeometry(0.029, 0.0055, 8, 28);
    bezel.rotateY(Math.PI / 2);
    bezel.translate(sx * 0.0445, 0, 0);
    b.C.push(bezel);

    const ledRing = new THREE.TorusGeometry(0.029, 0.0036, 10, 28);
    ledRing.rotateY(Math.PI / 2);
    ledRing.translate(sx * 0.0472, 0, 0);
    b.E.push(ledRing);
  }

  // Front ankle status arc
  const frontArc = new THREE.TorusGeometry(0.036, 0.0032, 8, 18, Math.PI * 0.55);
  frontArc.rotateX(Math.PI / 2);
  frontArc.rotateY(Math.PI * 0.22);
  frontArc.translate(0, -0.028, 0.024);
  b.E.push(frontArc);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 11. Foot Assembly (Tier 2, parent: ankle; athletic sneaker mechanical foot)
//     Ground sits at local Y = -0.140; toe hinge sits at [0, -0.105, +0.140]
// ---------------------------------------------------------------------------
function buildFootAssembly(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // --- Dark shock-absorbing sculpted sole (resting cleanly at Y = -0.1395) ---
  // Rear heel traction block (z: -0.095 .. -0.005)
  const heelSole = new THREE.BoxGeometry(0.114, 0.032, 0.092);
  heelSole.translate(0.0, -0.1235, -0.048);
  b.C.push(heelSole);

  // Mid-foot raised arch chassis bridge (creates athletic running-shoe cutout!)
  const archBridge = new THREE.BoxGeometry(0.096, 0.024, 0.080);
  archBridge.translate(0.0, -0.116, 0.024);
  b.C.push(archBridge);

  // Forefoot ball sole pad (z: +0.056 .. +0.138)
  const ballSole = new THREE.BoxGeometry(0.122, 0.032, 0.084);
  ballSole.translate(0.0, -0.1235, 0.096);
  b.C.push(ballSole);

  // Dark midsole chassis body
  const midChassis = new THREE.BoxGeometry(0.104, 0.048, 0.205);
  midChassis.translate(0.0, -0.092, 0.026);
  b.C.push(midChassis);

  // --- Cool-grey midsole side transition rails & heel stabilizer ---
  for (const sx of [-1, 1] as const) {
    const rail = new THREE.BoxGeometry(0.014, 0.028, 0.190);
    rail.translate(sx * 0.056, -0.098, 0.028);
    b.B.push(rail);
  }
  const heelBumper = new THREE.BoxGeometry(0.108, 0.044, 0.030);
  heelBumper.rotateX(-0.18);
  heelBumper.translate(0.0, -0.098, -0.084);
  b.B.push(heelBumper);

  // --- Pearl-white sculpted sneaker upper vamp & aerodynamic instep armor ---
  const vampPlate = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.048, 0.088),
      new THREE.Vector2(-0.056, -0.010),
      new THREE.Vector2(-0.046, -0.088),
      new THREE.Vector2(0.046, -0.088),
      new THREE.Vector2(0.056, -0.010),
      new THREE.Vector2(0.048, 0.088),
    ],
    {
      thickness: 0.018,
      bevel: 0.006,
      bendRadiusX: 0.085,
      bendRadiusY: 0.35,
      segments: 8,
      crownZ: 0.014,
    },
  );
  // Tilt forward like a running shoe vamp sweeping from ankle to forefoot
  vampPlate.rotateX(-Math.PI * 0.36);
  vampPlate.translate(0.0, -0.068, 0.058);
  b.A.push(vampPlate);

  // White sculpted heel counter shell wrapping rear of ankle
  const heelShell = createLoftedArmorShell(
    [
      { y: -0.106, centerX: 0.0, centerZ: -0.032, rx: 0.058, rz: 0.062, squareness: 0.32, angleStart: -Math.PI * 0.98, angleEnd: -Math.PI * 0.02 },
      { y: -0.064, centerX: 0.0, centerZ: -0.028, rx: 0.056, rz: 0.058, squareness: 0.32, angleStart: -Math.PI * 0.96, angleEnd: -Math.PI * 0.04 },
      { y: -0.028, centerX: 0.0, centerZ: -0.022, rx: 0.050, rz: 0.048, squareness: 0.28, angleStart: -Math.PI * 0.90, angleEnd: -Math.PI * 0.10 },
    ],
    14,
    0.011,
    false,
  );
  b.A.push(heelShell);

  // Outer-side shoe blue LED accent notch (matches SIDE view!)
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.061, -0.076, -0.022),
    new THREE.Vector3(0.063, -0.100, 0.046),
    new THREE.Vector3(1, 0, 0),
    0.005,
    0.012,
  );

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 11b. Articulated Sneaker Toe Cap (mounted on toe_R/L @ [0, -0.105, +0.140])
//      Ground sits at local Y = -0.035; spans z: 0.00 .. +0.088
// ---------------------------------------------------------------------------
export function buildDetailedToeCap(
  side: Side,
  materials: MaterialBank,
): THREE.Object3D {
  const b = createBuckets();

  // Flexible dark forefoot rubber sole (bottom at Y = -0.0345)
  const toeSole = new THREE.BoxGeometry(0.118, 0.018, 0.082);
  toeSole.translate(0.0, -0.0255, 0.040);
  b.C.push(toeSole);

  // Cool-grey toe bumper edge
  const bumper = new THREE.CylinderGeometry(0.055, 0.058, 0.020, 16, 1, false, -Math.PI * 0.5, Math.PI);
  bumper.scale(1.0, 1.0, 0.72);
  bumper.translate(0.0, -0.020, 0.054);
  b.B.push(bumper);

  // Sculpted pearl-white rounded toe-box armor flowing cleanly from vamp
  const toeUpper = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.056, -0.042),
      new THREE.Vector2(0.056, -0.042),
      new THREE.Vector2(0.048, 0.028),
      new THREE.Vector2(0.024, 0.048),
      new THREE.Vector2(-0.024, 0.048),
      new THREE.Vector2(-0.048, 0.028),
    ],
    {
      thickness: 0.018,
      bevel: 0.006,
      bendRadiusX: 0.080,
      bendRadiusY: 0.16,
      segments: 8,
      crownZ: 0.012,
    },
  );
  toeUpper.rotateX(-Math.PI * 0.34);
  toeUpper.translate(0.0, 0.008, 0.030);
  b.A.push(toeUpper);

  return finalizeBuckets(`foot_assembly_${side}`, 2, side, b, materials);
}

/**
 * Registers detailed leg builders for the specified side(s).
 * In M2 vertical slice: pass ['R'] to build only the right leg first;
 * after user approval, pass ['L', 'R'] to mirror the left leg cleanly.
 */
export function registerLegBuilders(
  registry: PartRegistry,
  sides: readonly Side[] = ['R'],
): void {
  const builders: ReadonlyArray<readonly [string, (e: PartCatalogEntry, m: MaterialBank) => THREE.Object3D]> = [
    ['hip_actuator', buildHipActuator],
    ['thigh_front_armor', buildThighFrontArmor],
    ['thigh_rear_armor', buildThighRearArmor],
    ['femur_frame', buildFemurFrame],
    ['knee_patella_guard', buildKneePatellaGuard],
    ['knee_drum_actuator', buildKneeDrumActuator],
    ['shin_front_armor', buildShinFrontArmor],
    ['calf_rear_armor', buildCalfRearArmor],
    ['tibia_frame', buildTibiaFrame],
    ['ankle_actuator', buildAnkleActuator],
    ['foot_assembly', buildFootAssembly],
  ];

  for (const side of sides) {
    for (const [baseId, builder] of builders) {
      registry.register(`${baseId}_${side}`, builder);
    }
    registry.registerToe(side, buildDetailedToeCap);
  }
}
