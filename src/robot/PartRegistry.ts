import * as THREE from 'three';
import type { MaterialBank } from '../config/materials.ts';
import type { PartCatalogEntry } from '../config/partsCatalog.ts';
import type { Side } from '../config/rig.ts';

export type PartBuilder = (entry: PartCatalogEntry, materials: MaterialBank) => THREE.Object3D;
export type ToeBuilder = (side: Side, materials: MaterialBank) => THREE.Object3D;
export interface HandFingerMeshes {
  phalangesC: THREE.InstancedMesh;
  padsA: THREE.InstancedMesh;
}
export type HandFingersBuilder = (side: Side, materials: MaterialBank) => HandFingerMeshes;

interface PlaceholderSpec {
  shape: 'box' | 'capsule' | 'cylinder';
  /** [widthX, heightY, depthZ] for box; [radius, totalLength, 0] for capsule/cylinder */
  size: readonly [number, number, number];
  /** Offset in parent joint's local frame (for left side; X auto-mirrored for right side) */
  offsetL: readonly [number, number, number];
  /** Optional cylinder/capsule axis orientation */
  axis?: 'x' | 'y' | 'z';
}

/**
 * M1 placeholder dimensions aligned to Section 2.2 proportions and Section 5.1 joint bind positions.
 * All offsets are in parentJoint's local coordinate system (L side; R mirrors X).
 */
const PLACEHOLDER_SPECS: Record<string, PlaceholderSpec> = {
  // Head (parent: head @ [0, 1.620, 0]; head spans Y 1.605..1.800)
  head_helmet_shell: { shape: 'capsule', size: [0.076, 0.185, 0], offsetL: [0, 0.088, -0.012] },
  head_visor: { shape: 'box', size: [0.118, 0.120, 0.048], offsetL: [0, 0.074, 0.068] },
  head_ai_core: { shape: 'box', size: [0.075, 0.070, 0.075], offsetL: [0, 0.085, -0.005] },
  head_ear_pod: { shape: 'cylinder', size: [0.032, 0.030, 0], offsetL: [0.084, 0.082, -0.006], axis: 'x' },

  // Neck (parent: neck @ [0, 1.545, -0.005]; spans Y 1.545..1.605)
  neck_actuator: { shape: 'cylinder', size: [0.042, 0.085, 0], offsetL: [0, 0.040, 0.002] },

  // Chest & Torso (parent: spine_upper @ [0, 1.370, -0.010]; chest Y 1.370..1.560)
  chest_front_armor: { shape: 'box', size: [0.340, 0.185, 0.105], offsetL: [0, 0.095, 0.078] },
  spine_energy_bar: { shape: 'box', size: [0.062, 0.180, 0.045], offsetL: [0, 0.092, -0.098] },
  torso_battery_pack: { shape: 'box', size: [0.165, 0.145, 0.080], offsetL: [0, 0.078, 0.015] },
  torso_compute_board: { shape: 'box', size: [0.145, 0.115, 0.040], offsetL: [0, 0.100, -0.045] },
  torso_main_chassis: { shape: 'box', size: [0.230, 0.165, 0.120], offsetL: [0, 0.085, 0.000] },
  back_scapula_armor: { shape: 'box', size: [0.125, 0.175, 0.065], offsetL: [0.105, 0.098, -0.082] },

  // Abdomen (parent: spine_lower @ [0, 1.180, -0.010]; spans Y 1.175..1.370)
  abdomen_hex_plate: { shape: 'box', size: [0.185, 0.175, 0.070], offsetL: [0, 0.095, 0.052] },
  lumbar_actuator: { shape: 'cylinder', size: [0.065, 0.175, 0], offsetL: [0, 0.092, -0.018] },

  // Pelvis (parent: pelvis @ [0, 1.050, 0]; spans Y 0.960..1.175)
  pelvis_front_armor: { shape: 'box', size: [0.190, 0.175, 0.080], offsetL: [0, 0.012, 0.064] },
  pelvis_rear_armor: { shape: 'box', size: [0.110, 0.155, 0.065], offsetL: [0.068, 0.015, -0.068] },
  pelvis_gimbal_base: { shape: 'box', size: [0.175, 0.115, 0.110], offsetL: [0, 0.015, -0.002] },
  hip_actuator: { shape: 'cylinder', size: [0.056, 0.078, 0], offsetL: [0.125, -0.008, 0.000], axis: 'x' },

  // Shoulder & Upper Arm (parent: pauldron_L / shoulder_L @ [0.205, 1.480, -0.010]; elbow @ [0.260, 1.250, -0.020])
  shoulder_pauldron: { shape: 'box', size: [0.115, 0.135, 0.135], offsetL: [0.052, 0.028, 0.005] },
  shoulder_actuator: { shape: 'cylinder', size: [0.048, 0.072, 0], offsetL: [0.010, 0.000, 0.000], axis: 'x' },
  upper_arm_armor: { shape: 'capsule', size: [0.048, 0.195, 0], offsetL: [0.030, -0.115, 0.000] },
  humerus_frame: { shape: 'cylinder', size: [0.024, 0.215, 0], offsetL: [0.026, -0.115, -0.005] },

  // Elbow & Forearm (parent: elbow_L @ [0.260, 1.250, -0.020]; wrist @ [0.290, 0.970, 0])
  elbow_actuator: { shape: 'cylinder', size: [0.040, 0.068, 0], offsetL: [0.000, 0.000, 0.000], axis: 'x' },
  forearm_gauntlet: { shape: 'capsule', size: [0.044, 0.235, 0], offsetL: [0.016, -0.140, 0.012] },
  forearm_frame: { shape: 'cylinder', size: [0.022, 0.250, 0], offsetL: [0.014, -0.140, 0.008] },

  // Hand (parent: wrist_L @ [0.290, 0.970, 0]; spans Y 0.800..0.970)
  hand_assembly: { shape: 'box', size: [0.042, 0.155, 0.082], offsetL: [0.006, -0.082, 0.005] },

  // Thigh (parent: hip_L @ [0.110, 1.040, 0]; knee @ [0.150, 0.660, 0.010])
  thigh_front_armor: { shape: 'capsule', size: [0.072, 0.340, 0], offsetL: [0.028, -0.185, 0.036] },
  thigh_rear_armor: { shape: 'capsule', size: [0.062, 0.320, 0], offsetL: [0.016, -0.185, -0.034] },
  femur_frame: { shape: 'cylinder', size: [0.032, 0.355, 0], offsetL: [0.020, -0.190, 0.005] },

  // Knee & Shin (parent: patella_L / knee_L @ [0.150, 0.660, 0.010]; ankle @ [0.190, 0.140, -0.020])
  knee_patella_guard: { shape: 'box', size: [0.082, 0.110, 0.048], offsetL: [0.004, 0.008, 0.078] },
  knee_drum_actuator: { shape: 'cylinder', size: [0.056, 0.108, 0], offsetL: [0.000, 0.000, 0.000], axis: 'x' },
  shin_front_armor: { shape: 'capsule', size: [0.058, 0.450, 0], offsetL: [0.024, -0.255, 0.024] },
  calf_rear_armor: { shape: 'capsule', size: [0.056, 0.415, 0], offsetL: [0.018, -0.245, -0.036] },
  tibia_frame: { shape: 'cylinder', size: [0.028, 0.485, 0], offsetL: [0.020, -0.260, -0.012] },

  // Ankle & Foot (parent: ankle_L @ [0.190, 0.140, -0.020]; ground Y=0 => local Y=-0.140)
  ankle_actuator: { shape: 'cylinder', size: [0.044, 0.088, 0], offsetL: [0.000, 0.000, 0.000], axis: 'x' },
  foot_assembly: { shape: 'box', size: [0.120, 0.095, 0.220], offsetL: [0.000, -0.092, 0.025] },
};

function createPlaceholderGeometry(spec: PlaceholderSpec): THREE.BufferGeometry {
  let geo: THREE.BufferGeometry;
  if (spec.shape === 'box') {
    geo = new THREE.BoxGeometry(spec.size[0], spec.size[1], spec.size[2]);
  } else if (spec.shape === 'capsule') {
    const r = spec.size[0];
    const cylLen = Math.max(0.01, spec.size[1] - 2 * r);
    geo = new THREE.CapsuleGeometry(r, cylLen, 6, 12);
  } else {
    geo = new THREE.CylinderGeometry(spec.size[0], spec.size[0], spec.size[1], 16);
  }

  if (spec.axis === 'x') {
    geo.rotateZ(Math.PI / 2);
  } else if (spec.axis === 'z') {
    geo.rotateX(Math.PI / 2);
  }
  return geo;
}

export function buildPlaceholderPart(
  entry: PartCatalogEntry,
  materials: MaterialBank,
): THREE.Object3D {
  const group = new THREE.Group();
  group.name = `part_mesh_${entry.partId}`;

  const spec = PLACEHOLDER_SPECS[entry.baseId] ?? {
    shape: 'box',
    size: [0.08, 0.08, 0.08],
    offsetL: [0, 0, 0],
  };

  const signX = entry.side === 'R' ? -1 : 1;
  const geo = createPlaceholderGeometry(spec);

  // Pick tier placeholder material (with obsidian visor / LED accent cues where helpful)
  let mat: THREE.Material = materials.tierPlaceholder[entry.tier];
  if (entry.baseId === 'head_visor') {
    mat = materials.grades.D;
  }

  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(
    spec.offsetL[0] * signX,
    spec.offsetL[1],
    spec.offsetL[2],
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData.partId = entry.partId;
  mesh.userData.tier = entry.tier;
  group.add(mesh);

  // Subtle LED indicator strip for parts that include Grade E in catalog (helps orientation check in M1)
  if (entry.materials.includes('E')) {
    const ledGeo = new THREE.BoxGeometry(
      Math.min(0.014, spec.size[0] * 0.25),
      Math.min(0.08, spec.size[1] * 0.45),
      0.008,
    );
    const led = new THREE.Mesh(ledGeo, materials.grades.E);
    const zPush = spec.shape === 'box' ? spec.size[2] * 0.5 + 0.002 : spec.size[0] + 0.002;
    const isRear =
      entry.baseId.includes('rear') ||
      entry.baseId.includes('calf') ||
      entry.baseId.includes('spine');
    led.position.set(
      spec.offsetL[0] * signX,
      spec.offsetL[1],
      spec.offsetL[2] + (isRear ? -zPush : zPush),
    );
    led.userData.partId = entry.partId;
    led.userData.tier = entry.tier;
    led.userData.isLed = true;
    group.add(led);
  }

  return group;
}

export class PartRegistry {
  public useGlbOverrides = false;
  private readonly builders = new Map<string, PartBuilder>();
  private readonly toeBuilders = new Map<Side, ToeBuilder>();
  private readonly handFingersBuilders = new Map<Side, HandFingersBuilder>();
  private readonly glbOverrides = new Map<string, THREE.Object3D>();

  public register(partOrBaseId: string, builder: PartBuilder): void {
    this.builders.set(partOrBaseId, builder);
  }

  public registerToe(side: Side, builder: ToeBuilder): void {
    this.toeBuilders.set(side, builder);
  }

  public registerHandFingers(side: Side, builder: HandFingersBuilder): void {
    this.handFingersBuilders.set(side, builder);
  }

  public registerGlbOverride(partId: string, model: THREE.Object3D): void {
    model.name = `glb_override_${partId}`;
    model.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
        const emissive =
          mat && 'emissiveIntensity' in mat
            ? (mat as THREE.MeshStandardMaterial).emissiveIntensity
            : 0;
        const isLed = emissive > 1.5;
        obj.castShadow = !isLed;
        obj.receiveShadow = !isLed;
        if (isLed) {
          obj.userData.isLed = true;
          obj.userData.materialGrade = 'E';
        }
      }
    });
    this.glbOverrides.set(partId, model);
  }

  public hasGlbOverride(partId: string): boolean {
    return this.glbOverrides.has(partId);
  }

  public clearGlbOverrides(): void {
    this.glbOverrides.clear();
  }

  /**
   * Probes and loads optional local GLB overrides from `public/models/parts/<partId>.glb`
   * when `useGlbOverrides` is true. Missing files or HTML SPA fallbacks are silently skipped.
   */
  public async loadGlbOverrides(
    partIds: readonly string[],
    basePath = '/models/parts',
  ): Promise<string[]> {
    if (!this.useGlbOverrides || typeof fetch === 'undefined') {
      return [];
    }

    const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');
    const loader = new GLTFLoader();
    const loaded: string[] = [];

    await Promise.all(
      partIds.map(async (partId) => {
        const url = `${basePath}/${partId}.glb`;
        try {
          const head = await fetch(url, { method: 'HEAD' });
          const ctype = head.headers.get('content-type') ?? '';
          if (!head.ok || ctype.includes('text/html')) return;
          const gltf = await loader.loadAsync(url);
          this.registerGlbOverride(partId, gltf.scene);
          loaded.push(partId);
        } catch {
          // Optional override missing; keep procedural geometry without error
        }
      }),
    );

    return loaded;
  }

  public build(entry: PartCatalogEntry, materials: MaterialBank): THREE.Object3D {
    if (this.useGlbOverrides) {
      const glb = this.glbOverrides.get(entry.partId);
      if (glb) {
        return glb.clone(true);
      }
    }
    const custom = this.builders.get(entry.partId) ?? this.builders.get(entry.baseId);
    if (custom) {
      return custom(entry, materials);
    }
    return buildPlaceholderPart(entry, materials);
  }

  public buildToe(side: Side, materials: MaterialBank): THREE.Object3D | null {
    const custom = this.toeBuilders.get(side);
    if (custom) {
      return custom(side, materials);
    }
    return null;
  }

  public buildHandFingers(side: Side, materials: MaterialBank): HandFingerMeshes | null {
    const custom = this.handFingersBuilders.get(side);
    if (custom) {
      return custom(side, materials);
    }
    return null;
  }
}
