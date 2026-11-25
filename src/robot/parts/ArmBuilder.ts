import * as THREE from 'three';
import type { MaterialBank } from '../../config/materials.ts';
import type { PartCatalogEntry } from '../../config/partsCatalog.ts';
import type { Side } from '../../config/rig.ts';
import {
  createConcentricDrumGeometry,
  createCurvedArmorPlate,
  createLoftedArmorShell,
  createStrutSegment,
  mergeGeometriesSafe,
  mirrorGeometryX,
} from '../GeometryHelpers.ts';
import type { HandFingerMeshes, PartRegistry } from '../PartRegistry.ts';
import {
  addRecessedLedStrip,
  createBuckets,
  createCurvedTube,
  finalizeBuckets,
} from './PartBuilderUtils.ts';

// Canonical LEFT arm (+X outward):
// shoulder_L @ [0.205, 1.480, -0.010] -> elbow_L @ [+0.055, -0.230, -0.010] local
// elbow_L    @ [0.260, 1.250, -0.020] -> wrist_L @ [+0.030, -0.280, +0.020] local
// wrist_L    @ [0.290, 0.970,  0.000] -> hand spans Y -0.170..0.000

// ---------------------------------------------------------------------------
// 1. Shoulder Pauldron (Tier 1, parent: pauldron_L/R; floating deltoid armor + inner blue LED!)
// ---------------------------------------------------------------------------
function buildShoulderPauldron(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Sculpted floating pearl-white deltoid shoulder shell (total span w/ pauldrons ≈ 0.58 m)
  const cap = createLoftedArmorShell(
    [
      { y: -0.080, centerX: 0.025, centerZ: 0.002, rx: 0.058, rz: 0.064, squareness: 0.35, angleStart: -Math.PI * 0.48, angleEnd: Math.PI * 0.48 },
      { y: -0.015, centerX: 0.022, centerZ: 0.002, rx: 0.076, rz: 0.074, squareness: 0.35, angleStart: -Math.PI * 0.52, angleEnd: Math.PI * 0.52 },
      { y: 0.045, centerX: 0.012, centerZ: 0.000, rx: 0.074, rz: 0.068, squareness: 0.32, angleStart: -Math.PI * 0.50, angleEnd: Math.PI * 0.50 },
      { y: 0.076, centerX: -0.006, centerZ: 0.000, rx: 0.052, rz: 0.048, squareness: 0.28, angleStart: -Math.PI * 0.42, angleEnd: Math.PI * 0.42 },
    ],
    18,
    0.013,
    false,
  );
  b.A.push(cap);

  // Cool-grey upper deltoid fairing crest (Grade B)
  const crest = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.026, -0.052),
      new THREE.Vector2(0.026, -0.052),
      new THREE.Vector2(0.020, 0.048),
      new THREE.Vector2(-0.020, 0.048),
    ],
    { thickness: 0.010, bevel: 0.003, bendRadiusX: 0.09, segments: 6 },
  );
  crest.rotateY(Math.PI * 0.5);
  crest.rotateX(-0.22);
  crest.translate(0.088, 0.022, 0.002);
  b.B.push(crest);

  // Signature inner lower-edge electric-blue light guide (key FRONT & SIDE anchor!)
  const frontLed = createCurvedTube(
    [
      new THREE.Vector3(0.028, -0.068, 0.060),
      new THREE.Vector3(0.068, -0.072, 0.048),
      new THREE.Vector3(0.084, -0.074, 0.010),
    ],
    0.0045,
    14,
    8,
  );
  b.E.push(frontLed);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 2. Shoulder Actuator (Tier 2, parent: shoulder_L/R; 3-axis harmonic drive)
// ---------------------------------------------------------------------------
function buildShoulderActuator(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const drum = createConcentricDrumGeometry({
    outerRadius: 0.044,
    width: 0.072,
    innerHubRadius: 0.022,
    boltCount: 6,
    segments: 24,
  });
  b.C.push(drum);

  const sphereCore = new THREE.SphereGeometry(0.042, 18, 14);
  b.C.push(sphereCore);

  const collar = new THREE.TorusGeometry(0.038, 0.004, 8, 22);
  collar.rotateX(Math.PI / 2);
  collar.translate(0.010, -0.040, -0.002);
  b.B.push(collar);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 3. Upper Arm Armor (Tier 1, parent: shoulder_L/R; sculpted two-piece bicep sleeve)
// ---------------------------------------------------------------------------
function buildUpperArmArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Outer-front & outer-rear white sleeve (leaving inner dark skeleton exposed as in reference!)
  const outerShell = createLoftedArmorShell(
    [
      { y: -0.195, centerX: 0.046, centerZ: -0.008, rx: 0.044, rz: 0.048, squareness: 0.30, angleStart: -Math.PI * 0.62, angleEnd: Math.PI * 0.62 },
      { y: -0.125, centerX: 0.032, centerZ: -0.005, rx: 0.052, rz: 0.056, squareness: 0.30, angleStart: -Math.PI * 0.68, angleEnd: Math.PI * 0.68 },
      { y: -0.055, centerX: 0.018, centerZ: -0.002, rx: 0.048, rz: 0.052, squareness: 0.28, angleStart: -Math.PI * 0.62, angleEnd: Math.PI * 0.62 },
    ],
    16,
    0.010,
    false,
  );
  b.A.push(outerShell);

  // Cool-grey mid-bicep accent plate
  const trim = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.022, -0.052),
      new THREE.Vector2(0.022, -0.046),
      new THREE.Vector2(0.020, 0.046),
      new THREE.Vector2(-0.020, 0.052),
    ],
    { thickness: 0.008, bevel: 0.003, bendRadiusX: 0.065, segments: 6 },
  );
  trim.rotateY(Math.PI * 0.5);
  trim.translate(0.082, -0.125, -0.004);
  b.B.push(trim);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 4. Humerus Frame (Tier 3, parent: shoulder_L/R; structural rod + cables)
// ---------------------------------------------------------------------------
function buildHumerusFrame(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  b.C.push(
    createStrutSegment(
      new THREE.Vector3(0.006, -0.025, -0.002),
      new THREE.Vector3(0.052, -0.218, -0.010),
      0.018,
      0.020,
      12,
    ),
  );

  // Inner auxiliary strut + copper power conduit
  b.F.push(
    createStrutSegment(
      new THREE.Vector3(-0.012, -0.042, 0.014),
      new THREE.Vector3(0.032, -0.205, 0.008),
      0.008,
      0.009,
      8,
    ),
  );

  const cable = createCurvedTube(
    [
      new THREE.Vector3(-0.006, -0.040, -0.020),
      new THREE.Vector3(0.016, -0.125, -0.028),
      new THREE.Vector3(0.040, -0.205, -0.022),
    ],
    0.0055,
    12,
    8,
  );
  b.copper.push(cable);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 5. Elbow Actuator (Tier 2, parent: elbow_L/R; exposed concentric drum)
// ---------------------------------------------------------------------------
function buildElbowActuator(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const drum = createConcentricDrumGeometry({
    outerRadius: 0.038,
    width: 0.068,
    innerHubRadius: 0.018,
    boltCount: 6,
    segments: 24,
  });
  b.C.push(drum);

  for (const sx of [-1, 1] as const) {
    const ring = new THREE.TorusGeometry(0.030, 0.0032, 8, 24);
    ring.rotateY(Math.PI / 2);
    ring.translate(sx * 0.036, 0, 0);
    b.B.push(ring);
  }

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 6. Forearm Gauntlet (Tier 1, parent: elbow_L/R; sleek sleeve + diagonal blue LED!)
// ---------------------------------------------------------------------------
function buildForearmGauntlet(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Sleek tapered pearl-white gauntlet from elbow (0,0,0) toward wrist (+0.030, -0.280, +0.020)
  const shell = createLoftedArmorShell(
    [
      { y: -0.245, centerX: 0.026, centerZ: 0.017, rx: 0.034, rz: 0.038, squareness: 0.34, angleStart: -Math.PI * 0.72, angleEnd: Math.PI * 0.72 },
      { y: -0.155, centerX: 0.018, centerZ: 0.012, rx: 0.044, rz: 0.048, squareness: 0.34, angleStart: -Math.PI * 0.78, angleEnd: Math.PI * 0.78 },
      { y: -0.055, centerX: 0.008, centerZ: 0.004, rx: 0.046, rz: 0.048, squareness: 0.32, angleStart: -Math.PI * 0.72, angleEnd: Math.PI * 0.72 },
    ],
    18,
    0.010,
    false,
  );
  b.A.push(shell);

  // Cool-grey wrist cuff collar
  const cuff = createLoftedArmorShell(
    [
      { y: -0.264, centerX: 0.028, centerZ: 0.019, rx: 0.033, rz: 0.036, squareness: 0.32, angleStart: -Math.PI * 0.65, angleEnd: Math.PI * 0.65 },
      { y: -0.240, centerX: 0.026, centerZ: 0.017, rx: 0.036, rz: 0.040, squareness: 0.32, angleStart: -Math.PI * 0.65, angleEnd: Math.PI * 0.65 },
    ],
    12,
    0.007,
    false,
  );
  b.B.push(cuff);

  // Signature outer diagonal electric-blue LED strip (key SIDE & FRONT view anchor!)
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0.052, -0.072, 0.020),
    new THREE.Vector3(0.060, -0.228, 0.034),
    new THREE.Vector3(0.92, 0, 0.39),
    0.0065,
    0.015,
  );

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 7. Forearm Frame (Tier 3, parent: elbow_L/R; twin-spar wrist linkage)
// ---------------------------------------------------------------------------
function buildForearmFrame(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Radius & ulna twin titanium spars
  b.C.push(
    createStrutSegment(
      new THREE.Vector3(0.004, -0.025, 0.012),
      new THREE.Vector3(0.030, -0.268, 0.026),
      0.012,
      0.014,
      10,
    ),
    createStrutSegment(
      new THREE.Vector3(0.002, -0.025, -0.014),
      new THREE.Vector3(0.028, -0.268, 0.008),
      0.011,
      0.013,
      10,
    ),
  );

  // Wrist gimbal collar
  const wristHub = new THREE.CylinderGeometry(0.024, 0.022, 0.034, 14);
  wristHub.translate(0.030, -0.268, 0.018);
  b.chrome.push(wristHub);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 8. Hand Assembly (Tier 2, parent: wrist_L/R; palm + dorsal white guard)
//    Articulated 5 fingers mount on finger_*_L/R inside hand_assembly container!
// ---------------------------------------------------------------------------
function buildHandAssembly(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Dark titanium mechanical palm chassis
  const palm = new THREE.BoxGeometry(0.028, 0.088, 0.068);
  palm.translate(0.002, -0.048, 0.004);
  b.C.push(palm);

  // Pearl-white sculpted dorsal hand armor plate (outer +X face of hanging hand)
  const dorsal = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.032, -0.042),
      new THREE.Vector2(0.032, -0.042),
      new THREE.Vector2(0.028, 0.038),
      new THREE.Vector2(-0.028, 0.038),
    ],
    { thickness: 0.009, bevel: 0.003, bendRadiusX: 0.075, segments: 6 },
  );
  dorsal.rotateY(Math.PI * 0.5);
  dorsal.translate(0.018, -0.048, 0.004);
  b.A.push(dorsal);

  // Cool-grey knuckle guard bar
  const knuckleBar = new THREE.BoxGeometry(0.032, 0.016, 0.070);
  knuckleBar.translate(0.004, -0.090, 0.004);
  b.B.push(knuckleBar);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

/**
 * Builds two InstancedMeshes (15 dark titanium phalanges + 10 pearl-white dorsal guards)
 * for all 5 articulated 3-phalanx fingers of one hand (reduces 30 draw calls -> 4 draw calls!).
 */
export function buildInstancedHandFingers(
  side: Side,
  materials: MaterialBank,
): HandFingerMeshes {
  // Unit phalanx geometry: origin at knuckle (0,0,0), extends along -Y to y = -1.0, radius = 1.0
  const cyl = new THREE.CylinderGeometry(0.95, 0.84, 0.88, 8);
  cyl.translate(0, -0.46, 0);
  const knuckle = new THREE.CylinderGeometry(0.98, 0.98, 0.14, 8);
  knuckle.translate(0, -0.02, 0);
  const unitPhalanxGeo = mergeGeometriesSafe([cyl, knuckle], true);

  let unitPadGeo: THREE.BufferGeometry = new THREE.BoxGeometry(0.64, 0.72, 1.56);
  unitPadGeo.translate(0.94, -0.48, 0);
  if (side === 'R') {
    const mirrored = mirrorGeometryX(unitPadGeo);
    unitPadGeo.dispose();
    unitPadGeo = mirrored;
  }

  const partId = `hand_assembly_${side}`;
  const phalangesC = new THREE.InstancedMesh(unitPhalanxGeo, materials.grades.C, 15);
  phalangesC.name = `fingers_phalanges_${side}`;
  phalangesC.castShadow = true;
  phalangesC.receiveShadow = true;
  phalangesC.userData.partId = partId;
  phalangesC.userData.tier = 2;
  phalangesC.userData.materialGrade = 'C';

  const padsA = new THREE.InstancedMesh(unitPadGeo, materials.grades.A, 10);
  padsA.name = `fingers_dorsal_pads_${side}`;
  padsA.castShadow = true;
  padsA.receiveShadow = true;
  padsA.userData.partId = partId;
  padsA.userData.tier = 2;
  padsA.userData.materialGrade = 'A';

  return { phalangesC, padsA };
}

export function registerArmBuilders(
  registry: PartRegistry,
  sides: readonly Side[] = ['L', 'R'],
): void {
  const builders: ReadonlyArray<
    readonly [string, (e: PartCatalogEntry, m: MaterialBank) => THREE.Object3D]
  > = [
    ['shoulder_pauldron', buildShoulderPauldron],
    ['shoulder_actuator', buildShoulderActuator],
    ['upper_arm_armor', buildUpperArmArmor],
    ['humerus_frame', buildHumerusFrame],
    ['elbow_actuator', buildElbowActuator],
    ['forearm_gauntlet', buildForearmGauntlet],
    ['forearm_frame', buildForearmFrame],
    ['hand_assembly', buildHandAssembly],
  ];

  for (const side of sides) {
    for (const [baseId, builder] of builders) {
      registry.register(`${baseId}_${side}`, builder);
    }
    registry.registerHandFingers(side, buildInstancedHandFingers);
  }
}
