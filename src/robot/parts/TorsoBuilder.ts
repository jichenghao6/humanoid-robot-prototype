import * as THREE from 'three';
import type { MaterialBank } from '../../config/materials.ts';
import type { PartCatalogEntry } from '../../config/partsCatalog.ts';
import {
  createCurvedArmorPlate,
  createHydraulicStrut,
  createStrutSegment,
} from '../GeometryHelpers.ts';
import type { PartRegistry } from '../PartRegistry.ts';
import {
  addRecessedLedStrip,
  createBuckets,
  createCurvedTube,
  finalizeBuckets,
} from './PartBuilderUtils.ts';

// ---------------------------------------------------------------------------
// 1. Neck Actuator (parent: neck @ [0, 1.545, -0.005]; spans Y 1.545..1.620)
// ---------------------------------------------------------------------------
function buildNeckActuator(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Central dark titanium multi-ring vertebral column
  const spineCol = new THREE.CylinderGeometry(0.032, 0.038, 0.088, 16);
  spineCol.translate(0, 0.042, -0.004);
  b.C.push(spineCol);

  for (let i = 0; i < 3; i++) {
    const ring = new THREE.TorusGeometry(0.038, 0.005, 8, 20);
    ring.rotateX(Math.PI / 2);
    ring.translate(0, 0.018 + i * 0.022, -0.004);
    b.F.push(ring);
  }

  // 4 mechanical tendon struts + curved cable bundles (Section 3.1.3)
  for (const sx of [-1, 1] as const) {
    const frontTendon = createHydraulicStrut(
      new THREE.Vector3(sx * 0.038, 0.004, 0.022),
      new THREE.Vector3(sx * 0.026, 0.080, 0.016),
      0.007,
      0.004,
    );
    b.C.push(frontTendon.barrel);
    b.chrome.push(frontTendon.rod);

    const rearTendon = createHydraulicStrut(
      new THREE.Vector3(sx * 0.036, 0.002, -0.028),
      new THREE.Vector3(sx * 0.024, 0.080, -0.024),
      0.007,
      0.004,
    );
    b.C.push(rearTendon.barrel);
    b.copper.push(rearTendon.rod);

    // Side braided neck cable
    const cable = createCurvedTube(
      [
        new THREE.Vector3(sx * 0.046, -0.005, 0.002),
        new THREE.Vector3(sx * 0.042, 0.038, -0.006),
        new THREE.Vector3(sx * 0.032, 0.078, -0.002),
      ],
      0.0045,
      14,
      8,
    );
    b.C.push(cable);
  }

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 2. Chest Front Armor (parent: spine_upper @ [0, 1.370, -0.010])
//    Heroic V-tapered broad chest puffed forward for natural S-curve + inverted triangle core ▽
// ---------------------------------------------------------------------------
function buildChestFrontArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Main broad sculpted pearl-white chest plate (forward S-curve projection at mid-chest)
  const mainPectoral = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.185, 0.088),
      new THREE.Vector2(-0.145, -0.025),
      new THREE.Vector2(-0.082, -0.095),
      new THREE.Vector2(0.082, -0.095),
      new THREE.Vector2(0.145, -0.025),
      new THREE.Vector2(0.185, 0.088),
      new THREE.Vector2(0.095, 0.118),
      new THREE.Vector2(-0.095, 0.118),
    ],
    {
      thickness: 0.024,
      bevel: 0.008,
      bendRadiusX: 0.34,
      bendRadiusY: 0.42,
      segments: 10,
      crownZ: 0.028,
    },
  );
  // Subtle forward chest puff (natural S-curve!)
  mainPectoral.rotateX(0.14);
  mainPectoral.translate(0, 0.092, 0.118);
  b.A.push(mainPectoral);

  // Upper collarbone trapezius armor bridges
  for (const sx of [-1, 1] as const) {
    const collar = createCurvedArmorPlate(
      [
        new THREE.Vector2(-0.055, -0.028),
        new THREE.Vector2(0.055, -0.028),
        new THREE.Vector2(0.048, 0.032),
        new THREE.Vector2(-0.042, 0.032),
      ],
      { thickness: 0.016, bevel: 0.005, bendRadiusX: 0.22, segments: 6 },
    );
    collar.rotateX(-Math.PI * 0.34);
    collar.rotateZ(sx * -0.14);
    collar.translate(sx * 0.105, 0.178, 0.028);
    b.A.push(collar);

    // Cool-grey pectoral lateral flank fairings (Grade B)
    const flank = createCurvedArmorPlate(
      [
        new THREE.Vector2(-0.036, -0.068),
        new THREE.Vector2(0.036, -0.058),
        new THREE.Vector2(0.042, 0.058),
        new THREE.Vector2(-0.032, 0.068),
      ],
      { thickness: 0.012, bevel: 0.004, bendRadiusX: 0.16, segments: 6 },
    );
    flank.rotateY(sx * Math.PI * 0.34);
    flank.translate(sx * 0.156, 0.072, 0.060);
    b.B.push(flank);
  }

  // Dark central recessed housing for the inverted-triangle (▽) energy core
  const triRecessShape = new THREE.Shape([
    new THREE.Vector2(-0.036, 0.022),
    new THREE.Vector2(0.036, 0.022),
    new THREE.Vector2(0.0, -0.032),
  ]);
  const triRecess = new THREE.ExtrudeGeometry(triRecessShape, { depth: 0.012, bevelEnabled: false });
  triRecess.rotateX(0.14);
  triRecess.translate(0, 0.112, 0.142);
  b.C.push(triRecess);

  // Outer glowing electric-blue inverted-triangle rim (▽) — key FRONT view anchor!
  const vTopL = new THREE.Vector3(-0.028, 0.130, 0.155);
  const vTopR = new THREE.Vector3(0.028, 0.130, 0.155);
  const vBot = new THREE.Vector3(0.0, 0.088, 0.161);
  b.E.push(
    createCurvedTube([vTopL, vTopR], 0.0042, 6, 8),
    createCurvedTube([vTopR, vBot], 0.0042, 6, 8),
    createCurvedTube([vBot, vTopL], 0.0042, 6, 8),
  );

  // Concentric inner core inverted triangle (▽) seated cleanly inside the rim
  const innerShape = new THREE.Shape([
    new THREE.Vector2(-0.015, 0.009),
    new THREE.Vector2(0.015, 0.009),
    new THREE.Vector2(0.0, -0.014),
  ]);
  const innerCore = new THREE.ExtrudeGeometry(innerShape, { depth: 0.004, bevelEnabled: false });
  innerCore.rotateX(0.14);
  innerCore.translate(0, 0.113, 0.153);
  b.E.push(innerCore);

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 3. Back Scapula Armor (paired _L/_R, parent: spine_upper)
//    Symmetrical wing-shaped white scapula plates flanking vertical spine core
// ---------------------------------------------------------------------------
function buildBackScapulaArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Canonical L wing-shaped scapula plate (x > 0, rear -Z, angled outward & slightly flared back for S-curve)
  const wing = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.042, 0.102),
      new THREE.Vector2(0.068, 0.084),
      new THREE.Vector2(0.078, -0.010),
      new THREE.Vector2(0.036, -0.098),
      new THREE.Vector2(-0.034, -0.082),
      new THREE.Vector2(-0.050, 0.020),
    ],
    {
      thickness: 0.020,
      bevel: 0.006,
      bendRadiusX: 0.20,
      bendRadiusY: 0.34,
      segments: 8,
      crownZ: 0.014,
    },
  );
  // Flip to face rearward (-Z) and flare gently for upper-back S-curve
  wing.rotateY(Math.PI - 0.24);
  wing.rotateX(0.12);
  wing.translate(0.104, 0.096, -0.096);
  b.A.push(wing);

  // Cool-grey upper-back exhaust fairing fin
  const ventTrim = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.024, -0.045),
      new THREE.Vector2(0.026, -0.035),
      new THREE.Vector2(0.022, 0.045),
      new THREE.Vector2(-0.024, 0.035),
    ],
    { thickness: 0.010, bevel: 0.003, bendRadiusX: 0.14, segments: 6 },
  );
  ventTrim.rotateY(Math.PI - 0.30);
  ventTrim.translate(0.118, 0.018, -0.092);
  b.B.push(ventTrim);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 4. Spine Energy Bar (Tier 2, parent: spine_upper; BACK view vertical blue core!)
// ---------------------------------------------------------------------------
function buildSpineEnergyBar(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Dark titanium dorsal spine housing
  const housing = new THREE.BoxGeometry(0.058, 0.195, 0.044);
  housing.rotateX(0.10);
  housing.translate(0, 0.094, -0.098);
  b.C.push(housing);

  // Segmented vertebral armor ribs along spine
  for (let i = 0; i < 4; i++) {
    const rib = new THREE.BoxGeometry(0.070, 0.014, 0.022);
    rib.translate(0, 0.020 + i * 0.048, -0.106 - i * 0.004);
    b.F.push(rib);
  }

  // Prominent vertical electric-blue spine energy core bar (key BACK view anchor!)
  addRecessedLedStrip(
    b,
    new THREE.Vector3(0, 0.172, -0.124),
    new THREE.Vector3(0, 0.018, -0.114),
    new THREE.Vector3(0, 0, -1),
    0.014,
    0.028,
  );

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 5. Torso Battery Pack (Tier 3, parent: spine_upper; 2.3 kWh solid-state pack)
// ---------------------------------------------------------------------------
function buildTorsoBatteryPack(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const pack = new THREE.BoxGeometry(0.160, 0.135, 0.072);
  pack.translate(0, 0.078, 0.024);
  b.F.push(pack);

  // 4 copper solid-state cell busbars & thermal plates
  for (let i = 0; i < 4; i++) {
    const cell = new THREE.BoxGeometry(0.032, 0.124, 0.078);
    cell.translate(-0.054 + i * 0.036, 0.078, 0.024);
    b.copper.push(cell);
  }

  // Retention rails
  const railTop = new THREE.BoxGeometry(0.172, 0.016, 0.082);
  railTop.translate(0, 0.145, 0.024);
  const railBot = new THREE.BoxGeometry(0.172, 0.016, 0.082);
  railBot.translate(0, 0.012, 0.024);
  b.C.push(railTop, railBot);

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 6. Torso Compute Board (Tier 3, parent: spine_upper; motherboard + copper heatpipes)
// ---------------------------------------------------------------------------
function buildTorsoComputeBoard(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const board = new THREE.BoxGeometry(0.150, 0.118, 0.022);
  board.translate(0, 0.098, -0.044);
  b.F.push(board);

  // Heatsink processor blocks
  for (const sx of [-1, 1] as const) {
    const cpu = new THREE.BoxGeometry(0.044, 0.044, 0.018);
    cpu.translate(sx * 0.036, 0.104, -0.056);
    b.chrome.push(cpu);

    // Curved copper heatpipes routing to rear radiator
    const pipe = createCurvedTube(
      [
        new THREE.Vector3(sx * 0.036, 0.118, -0.058),
        new THREE.Vector3(sx * 0.058, 0.098, -0.068),
        new THREE.Vector3(sx * 0.042, 0.052, -0.062),
      ],
      0.006,
      14,
      8,
    );
    b.copper.push(pipe);
  }

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 7. Torso Main Chassis (Tier 3 anchor, parent: spine_upper; titanium ribcage)
// ---------------------------------------------------------------------------
function buildTorsoMainChassis(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Central spine column & upper shoulder cross-beam
  const spine = new THREE.CylinderGeometry(0.032, 0.036, 0.190, 12);
  spine.translate(0, 0.092, -0.010);
  b.C.push(spine);

  const shoulderBeam = new THREE.BoxGeometry(0.360, 0.042, 0.068);
  shoulderBeam.translate(0, 0.112, 0.0);
  b.C.push(shoulderBeam);

  // Structural ribcage side spars
  for (const sx of [-1, 1] as const) {
    b.C.push(
      createStrutSegment(
        new THREE.Vector3(sx * 0.165, 0.112, 0.0),
        new THREE.Vector3(sx * 0.090, 0.005, 0.0),
        0.016,
        0.020,
        10,
      ),
    );
    b.F.push(
      createStrutSegment(
        new THREE.Vector3(sx * 0.140, 0.140, 0.040),
        new THREE.Vector3(sx * 0.075, 0.010, 0.035),
        0.011,
        0.013,
        8,
      ),
    );
  }

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 8. Abdomen Hex Plate (Tier 1, parent: spine_lower @ [0, 1.180, -0.010])
//    Narrow athletic waist (S-curve pinch) + 3-segment dark hex abdomen + twin curved blue LEDs!
// ---------------------------------------------------------------------------
function buildAbdomenHexPlate(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Three articulated hexagonal dark-titanium abdominal plates (FRONT view anchor!)
  // Bottom tier extends slightly below spine_lower origin to tuck cleanly behind pelvis belt
  const tiers = [
    { y: 0.155, w: 0.078, h: 0.032, z: 0.088 },
    { y: 0.092, w: 0.068, h: 0.034, z: 0.078 },
    { y: 0.026, w: 0.056, h: 0.034, z: 0.072 },
  ] as const;

  for (const t of tiers) {
    const hex = createCurvedArmorPlate(
      [
        new THREE.Vector2(-t.w, 0),
        new THREE.Vector2(-t.w * 0.62, -t.h),
        new THREE.Vector2(t.w * 0.62, -t.h),
        new THREE.Vector2(t.w, 0),
        new THREE.Vector2(t.w * 0.62, t.h),
        new THREE.Vector2(-t.w * 0.62, t.h),
      ],
      { thickness: 0.014, bevel: 0.004, bendRadiusX: 0.16, segments: 6, crownZ: 0.006 },
    );
    hex.translate(0, t.y, t.z);
    b.C.push(hex);
  }

  // Dark waist rib flank panels reaching down to meet the pelvic upper rim
  for (const sx of [-1, 1] as const) {
    const oblique = createCurvedArmorPlate(
      [
        new THREE.Vector2(-0.028, -0.108),
        new THREE.Vector2(0.032, -0.094),
        new THREE.Vector2(0.038, 0.075),
        new THREE.Vector2(-0.026, 0.082),
      ],
      { thickness: 0.012, bevel: 0.003, bendRadiusX: 0.13, segments: 6 },
    );
    oblique.rotateY(sx * Math.PI * 0.38);
    oblique.translate(sx * 0.092, 0.095, 0.026);
    b.C.push(oblique);

    // Iconic downward-and-inward curving electric-blue abdominal side light bands (FRONT & SIDE anchor!)
    const ledCurve = createCurvedTube(
      [
        new THREE.Vector3(sx * 0.112, 0.162, 0.056),
        new THREE.Vector3(sx * 0.096, 0.096, 0.068),
        new THREE.Vector3(sx * 0.076, 0.024, 0.070),
      ],
      0.0052,
      18,
      8,
    );
    b.E.push(ledCurve);
  }

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 9. Lumbar Actuator (Tier 2, parent: spine_lower; exposed rear spine + gimbal socket!)
// ---------------------------------------------------------------------------
function buildLumbarActuator(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Central dark lumbar column spanning cleanly from gimbal socket (-0.015) up into spine_upper (+0.192)
  const col = new THREE.CylinderGeometry(0.044, 0.050, 0.207, 18);
  col.translate(0, 0.0885, -0.004);
  b.C.push(col);

  // Lower gimbal socket collar & bearing rings centered at spine_lower pivot (0, 0, 0)
  const socketCollar = new THREE.CylinderGeometry(0.052, 0.056, 0.030, 20);
  socketCollar.translate(0, -0.001, 0.0);
  b.C.push(socketCollar);

  for (const ry of [-0.013, 0.012] as const) {
    const ring = new THREE.TorusGeometry(0.053, 0.0055, 10, 24);
    ring.rotateX(Math.PI / 2);
    ring.translate(0, ry, 0.0);
    b.F.push(ring);
  }

  // Exposed rear spinal linkage blocks (lowest vertebra bridges down to pelvic notch)
  const vertebraYs = [-0.014, 0.026, 0.068, 0.110, 0.154] as const;
  for (const vy of vertebraYs) {
    const vertebra = new THREE.BoxGeometry(0.046, 0.024, 0.044);
    vertebra.translate(0, vy, -0.056);
    b.F.push(vertebra);
  }

  for (const sx of [-1, 1] as const) {
    const damperStart = new THREE.Vector3(sx * 0.060, -0.018, -0.044);
    const damper = createHydraulicStrut(
      damperStart,
      new THREE.Vector3(sx * 0.078, 0.175, -0.058),
      0.012,
      0.007,
    );
    b.C.push(damper.barrel);
    b.chrome.push(damper.rod);

    // Lower damper spherical clevis joint seated on pelvis_gimbal_base rear boss
    const clevis = new THREE.SphereGeometry(0.013, 12, 10);
    clevis.translate(damperStart.x, damperStart.y, damperStart.z);
    b.C.push(clevis);

    // Symmetrical lower-back diagonal electric-blue LED bars (key BACK view anchor!)
    addRecessedLedStrip(
      b,
      new THREE.Vector3(sx * 0.096, 0.168, -0.068),
      new THREE.Vector3(sx * 0.058, 0.028, -0.064),
      new THREE.Vector3(sx * 0.35, 0, -0.94),
      0.0075,
      0.017,
    );
  }

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 10. Pelvis Front Armor (Tier 1, parent: pelvis @ [0, 1.050, 0]; V-shaped white guard!)
// ---------------------------------------------------------------------------
function buildPelvisFrontArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Iconic V-shaped pearl-white pelvic groin armor (key FRONT view anchor!)
  const vPlate = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.112, 0.098),
      new THREE.Vector2(-0.088, 0.015),
      new THREE.Vector2(-0.038, -0.082),
      new THREE.Vector2(0.038, -0.082),
      new THREE.Vector2(0.088, 0.015),
      new THREE.Vector2(0.112, 0.098),
      new THREE.Vector2(0.052, 0.114),
      new THREE.Vector2(-0.052, 0.114),
    ],
    {
      thickness: 0.020,
      bevel: 0.006,
      bendRadiusX: 0.19,
      bendRadiusY: 0.32,
      segments: 8,
      crownZ: 0.016,
    },
  );
  vPlate.rotateX(-0.08);
  vPlate.translate(0, 0.012, 0.086);
  b.A.push(vPlate);

  // Cool-grey upper belt trim wrapping cleanly around waist transition (World Y ~ 1.146..1.180)
  const belt = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.104, -0.016),
      new THREE.Vector2(0.104, -0.016),
      new THREE.Vector2(0.096, 0.018),
      new THREE.Vector2(-0.096, 0.018),
    ],
    { thickness: 0.012, bevel: 0.003, bendRadiusX: 0.15, segments: 6 },
  );
  belt.translate(0, 0.112, 0.086);
  b.B.push(belt);

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

// ---------------------------------------------------------------------------
// 11. Pelvis Rear Armor (paired _L/_R, parent: pelvis; split rear white plates)
// ---------------------------------------------------------------------------
function buildPelvisRearArmor(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  // Canonical L rear pelvic plate with raised outer crest cradling the waist flank
  const plate = createCurvedArmorPlate(
    [
      new THREE.Vector2(-0.048, 0.092),
      new THREE.Vector2(0.052, 0.078),
      new THREE.Vector2(0.044, -0.052),
      new THREE.Vector2(-0.018, -0.078),
      new THREE.Vector2(-0.054, -0.016),
    ],
    {
      thickness: 0.018,
      bevel: 0.005,
      bendRadiusX: 0.16,
      bendRadiusY: 0.28,
      segments: 8,
      crownZ: 0.012,
    },
  );
  plate.rotateY(Math.PI - 0.18);
  plate.translate(0.058, 0.022, -0.086);
  b.A.push(plate);

  return finalizeBuckets(entry.partId, entry.tier, entry.side ?? 'L', b, materials);
}

// ---------------------------------------------------------------------------
// 12. Pelvis Gimbal Base (Tier 3 anchor, parent: pelvis @ [0, 1.050, 0]; IMU & lumbar gimbal)
// ---------------------------------------------------------------------------
function buildPelvisGimbalBase(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
  const b = createBuckets();

  const core = new THREE.BoxGeometry(0.160, 0.105, 0.110);
  core.translate(0, 0.015, 0.0);
  b.C.push(core);

  // Horizontal hip axle shaft
  const axle = new THREE.CylinderGeometry(0.032, 0.032, 0.240, 16);
  axle.rotateZ(Math.PI / 2);
  axle.translate(0, -0.010, 0.0);
  b.F.push(axle);

  // Central IMU module (copper-shielded, seated slightly forward for X-Ray/explode legibility)
  const imu = new THREE.BoxGeometry(0.056, 0.042, 0.056);
  imu.translate(0, 0.058, 0.028);
  b.copper.push(imu);

  // Tapered upper pelvic pedestal bridging core (Y=1.106) up toward lumbar gimbal (Y=1.170)
  const pedestal = new THREE.CylinderGeometry(0.048, 0.076, 0.064, 20);
  pedestal.scale(1.18, 1.0, 0.88);
  pedestal.translate(0, 0.088, -0.008);
  b.C.push(pedestal);

  // Machined bearing flange & neck rising to spine_lower center (local [0, 0.130, -0.010])
  const flange = new THREE.CylinderGeometry(0.056, 0.064, 0.014, 20);
  flange.translate(0, 0.084, -0.010);
  b.F.push(flange);

  const gimbalNeck = new THREE.CylinderGeometry(0.036, 0.046, 0.034, 20);
  gimbalNeck.translate(0, 0.114, -0.010);
  b.F.push(gimbalNeck);

  // Spherical gimbal ball centered at spine_lower joint pivot (World [0, 1.180, -0.010])
  const gimbalBall = new THREE.SphereGeometry(0.044, 20, 16);
  gimbalBall.translate(0, 0.130, -0.010);
  b.F.push(gimbalBall);

  // Lateral pelvic flank gussets & rear lumbar damper lower mounting bosses
  for (const sx of [-1, 1] as const) {
    b.C.push(
      createStrutSegment(
        new THREE.Vector3(sx * 0.078, 0.040, 0.002),
        new THREE.Vector3(sx * 0.056, 0.104, -0.010),
        0.022,
        0.015,
        10,
      ),
    );
    b.C.push(
      createStrutSegment(
        new THREE.Vector3(sx * 0.056, 0.048, -0.040),
        new THREE.Vector3(sx * 0.060, 0.110, -0.054),
        0.016,
        0.012,
        10,
      ),
    );
    const boss = new THREE.SphereGeometry(0.014, 12, 10);
    boss.translate(sx * 0.060, 0.112, -0.054);
    b.F.push(boss);
  }

  return finalizeBuckets(entry.partId, entry.tier, null, b, materials);
}

export function registerTorsoBuilders(registry: PartRegistry): void {
  registry.register('neck_actuator', buildNeckActuator);
  registry.register('chest_front_armor', buildChestFrontArmor);
  registry.register('back_scapula_armor', buildBackScapulaArmor);
  registry.register('spine_energy_bar', buildSpineEnergyBar);
  registry.register('torso_battery_pack', buildTorsoBatteryPack);
  registry.register('torso_compute_board', buildTorsoComputeBoard);
  registry.register('torso_main_chassis', buildTorsoMainChassis);
  registry.register('abdomen_hex_plate', buildAbdomenHexPlate);
  registry.register('lumbar_actuator', buildLumbarActuator);
  registry.register('pelvis_front_armor', buildPelvisFrontArmor);
  registry.register('pelvis_rear_armor', buildPelvisRearArmor);
  registry.register('pelvis_gimbal_base', buildPelvisGimbalBase);
}
