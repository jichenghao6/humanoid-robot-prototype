import * as THREE from 'three';
import type { QualityLevel } from '../core/AppState.ts';
import type { MaterialGrade, PartTier } from './partsCatalog.ts';

export type MaterialGradeKey = MaterialGrade | 'copper' | 'chrome';

export interface MaterialBank {
  /** Placeholder tier-colored materials */
  tierPlaceholder: Record<PartTier, THREE.MeshStandardMaterial>;
  /** Six-grade PBR materials (A..F) */
  grades: Record<MaterialGrade, THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial>;
  /** Secondary copper accent for Grade F (heatpipes / pistons) */
  gradeFCopper: THREE.MeshStandardMaterial;
  /** Machined bright steel/chrome accent for hydraulic rods */
  gradeFChrome: THREE.MeshStandardMaterial;
  /** Pre-created shared hover/selection highlight variants (zero runtime allocations) */
  highlights: Record<MaterialGradeKey, THREE.Material>;
  /** Pre-created shared holographic Fresnel X-Ray material for Tier 1 shells */
  xrayShell: THREE.ShaderMaterial;
  /** Pre-created shared tier-colored wireframe materials */
  wireframe: Record<PartTier, THREE.MeshBasicMaterial>;
  /** Shared pure black material for selective bloom mask pass */
  darkMask: THREE.MeshBasicMaterial;
  /** Cached procedural textures */
  textures: {
    armorBump: THREE.CanvasTexture | null;
    armorRoughness: THREE.CanvasTexture | null;
    metalRoughness: THREE.CanvasTexture | null;
  };
  getBaseMaterial: (gradeKey: string) => THREE.Material;
  getHighlightMaterial: (gradeKey: string) => THREE.Material;
  setQuality: (quality: QualityLevel) => void;
  dispose: () => void;
}

const XRAY_VERTEX = /* glsl */ `
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  void main() {
    vec4 localPos = vec4(position, 1.0);
    vec3 transformedNormal = normal;
    #ifdef USE_INSTANCING
      localPos = instanceMatrix * localPos;
      mat3 instMat3 = mat3(instanceMatrix);
      transformedNormal /= vec3(
        max(1e-6, dot(instMat3[0], instMat3[0])),
        max(1e-6, dot(instMat3[1], instMat3[1])),
        max(1e-6, dot(instMat3[2], instMat3[2]))
      );
      transformedNormal = instMat3 * transformedNormal;
    #endif
    vec4 mvPos = modelViewMatrix * localPos;
    vNormalView = normalize(normalMatrix * transformedNormal);
    vViewDir = normalize(-mvPos.xyz);
    gl_Position = projectionMatrix * mvPos;
  }
`;

const XRAY_FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform float uPower;
  uniform float uIntensity;
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  void main() {
    float ndotv = abs(dot(normalize(vNormalView), normalize(vViewDir)));
    float fresnel = pow(1.0 - clamp(ndotv, 0.0, 1.0), uPower);
    float alpha = clamp(0.05 + fresnel * uIntensity, 0.0, 0.92);
    gl_FragColor = vec4(uColor * (0.28 + fresnel * 1.45), alpha);
  }
`;

function createProceduralTextures(): {
  armorBump: THREE.CanvasTexture | null;
  armorRoughness: THREE.CanvasTexture | null;
  metalRoughness: THREE.CanvasTexture | null;
} {
  if (typeof document === 'undefined') {
    return { armorBump: null, armorRoughness: null, metalRoughness: null };
  }

  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = 512;
  bumpCanvas.height = 512;
  const bCtx = bumpCanvas.getContext('2d')!;
  bCtx.fillStyle = '#808080';
  bCtx.fillRect(0, 0, 512, 512);
  bCtx.strokeStyle = '#3c3c3c';
  bCtx.lineWidth = 2;
  bCtx.strokeRect(24, 24, 464, 464);
  bCtx.beginPath();
  bCtx.moveTo(24, 256);
  bCtx.lineTo(488, 256);
  bCtx.moveTo(256, 24);
  bCtx.lineTo(256, 488);
  bCtx.stroke();
  bCtx.fillStyle = '#484848';
  for (const [fx, fy] of [
    [42, 42],
    [470, 42],
    [42, 470],
    [470, 470],
    [42, 240],
    [470, 240],
  ] as const) {
    bCtx.beginPath();
    bCtx.arc(fx, fy, 4, 0, Math.PI * 2);
    bCtx.fill();
  }
  const armorBump = new THREE.CanvasTexture(bumpCanvas);
  armorBump.wrapS = THREE.RepeatWrapping;
  armorBump.wrapT = THREE.RepeatWrapping;
  armorBump.repeat.set(2, 2);

  const roughCanvas = document.createElement('canvas');
  roughCanvas.width = 256;
  roughCanvas.height = 256;
  const rCtx = roughCanvas.getContext('2d')!;
  rCtx.fillStyle = '#808080';
  rCtx.fillRect(0, 0, 256, 256);
  let seed = 1337;
  const rand = (): number => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  for (let i = 0; i < 2400; i++) {
    const v = Math.floor(118 + rand() * 24);
    rCtx.fillStyle = `rgb(${v},${v},${v})`;
    rCtx.fillRect(rand() * 256, rand() * 256, 2, 2);
  }
  const armorRoughness = new THREE.CanvasTexture(roughCanvas);
  armorRoughness.wrapS = THREE.RepeatWrapping;
  armorRoughness.wrapT = THREE.RepeatWrapping;
  armorRoughness.repeat.set(3, 3);

  const metalCanvas = document.createElement('canvas');
  metalCanvas.width = 256;
  metalCanvas.height = 256;
  const mCtx = metalCanvas.getContext('2d')!;
  mCtx.fillStyle = '#848484';
  mCtx.fillRect(0, 0, 256, 256);
  for (let y = 0; y < 256; y++) {
    const v = Math.floor(112 + rand() * 34);
    mCtx.fillStyle = `rgb(${v},${v},${v})`;
    mCtx.fillRect(0, y, 256, 1);
  }
  const metalRoughness = new THREE.CanvasTexture(metalCanvas);
  metalRoughness.wrapS = THREE.RepeatWrapping;
  metalRoughness.wrapT = THREE.RepeatWrapping;
  metalRoughness.repeat.set(2, 2);

  return { armorBump, armorRoughness, metalRoughness };
}

function makeHighlightClone(
  src: THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial,
  emissiveHex: number,
  emissiveIntensity: number,
): THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial {
  const cloned = src.clone();
  cloned.name = `${src.name}_Highlight`;
  cloned.emissive = new THREE.Color(emissiveHex);
  cloned.emissiveIntensity = emissiveIntensity;
  return cloned;
}

export function createMaterialBank(initialQuality: QualityLevel = 'High'): MaterialBank {
  const textures = createProceduralTextures();

  const tierPlaceholder: Record<PartTier, THREE.MeshStandardMaterial> = {
    1: new THREE.MeshStandardMaterial({
      name: 'M1_Tier1_Armor',
      color: 0xdce4f0,
      metalness: 0.14,
      roughness: 0.34,
    }),
    2: new THREE.MeshStandardMaterial({
      name: 'M1_Tier2_Actuator',
      color: 0x3b495e,
      metalness: 0.72,
      roughness: 0.38,
    }),
    3: new THREE.MeshStandardMaterial({
      name: 'M1_Tier3_Core',
      color: 0xd48a4a,
      metalness: 0.78,
      roughness: 0.32,
    }),
  };

  const gradeA = new THREE.MeshPhysicalMaterial({
    name: 'GradeA_PearlWhite',
    color: 0xe6ecf5,
    metalness: 0.10,
    roughness: 0.30,
    clearcoat: initialQuality === 'High' ? 0.60 : 0.0,
    clearcoatRoughness: 0.20,
    bumpMap: textures.armorBump,
    bumpScale: 0.0018,
    roughnessMap: textures.armorRoughness,
  });

  const gradeB = new THREE.MeshPhysicalMaterial({
    name: 'GradeB_CoolGrey',
    color: 0xb9c4d2,
    metalness: 0.28,
    roughness: 0.36,
    clearcoat: initialQuality === 'High' ? 0.25 : 0.0,
    clearcoatRoughness: 0.28,
    roughnessMap: textures.armorRoughness,
  });

  const gradeC = new THREE.MeshPhysicalMaterial({
    name: 'GradeC_DarkTitanium',
    color: 0x1d222c,
    metalness: 0.80,
    roughness: 0.42,
    clearcoat: initialQuality === 'High' ? 0.10 : 0.0,
    clearcoatRoughness: 0.35,
    roughnessMap: textures.metalRoughness,
  });

  const gradeD = new THREE.MeshPhysicalMaterial({
    name: 'GradeD_ObsidianVisor',
    color: 0x060910,
    metalness: 0.15,
    roughness: 0.06,
    clearcoat: initialQuality === 'High' ? 1.00 : 0.2,
    clearcoatRoughness: 0.04,
  });

  const gradeE = new THREE.MeshStandardMaterial({
    name: 'GradeE_ElectricBlueLED',
    color: 0x4ea5f2,
    metalness: 0.0,
    roughness: 0.20,
    emissive: new THREE.Color(0x1f86ff),
    emissiveIntensity: 3.0,
  });

  const gradeF = new THREE.MeshStandardMaterial({
    name: 'GradeF_CoreAlloy',
    color: 0x3b4454,
    metalness: 0.85,
    roughness: 0.30,
    roughnessMap: textures.metalRoughness,
  });

  const gradeFCopper = new THREE.MeshStandardMaterial({
    name: 'GradeF_CopperHeatpipe',
    color: 0xd48a4a,
    metalness: 0.85,
    roughness: 0.30,
  });

  const gradeFChrome = new THREE.MeshStandardMaterial({
    name: 'GradeF_HydraulicChrome',
    color: 0xcbd5e4,
    metalness: 0.92,
    roughness: 0.18,
  });

  const highlights: Record<MaterialGradeKey, THREE.Material> = {
    A: makeHighlightClone(gradeA, 0x2a88ff, 0.42),
    B: makeHighlightClone(gradeB, 0x2a88ff, 0.48),
    C: makeHighlightClone(gradeC, 0x2a88ff, 0.65),
    D: makeHighlightClone(gradeD, 0x2a88ff, 0.55),
    E: gradeE,
    F: makeHighlightClone(gradeF, 0x2a88ff, 0.62),
    copper: makeHighlightClone(gradeFCopper, 0x4aa8ff, 0.55),
    chrome: makeHighlightClone(gradeFChrome, 0x2a88ff, 0.48),
  };

  // Order-independent holographic Fresnel X-Ray material (Section 4.4)
  const xrayShell = new THREE.ShaderMaterial({
    name: 'XRay_Fresnel_Tier1',
    uniforms: {
      uColor: { value: new THREE.Color(0x3ba8ff) },
      uPower: { value: 2.2 },
      uIntensity: { value: 0.92 },
    },
    vertexShader: XRAY_VERTEX,
    fragmentShader: XRAY_FRAGMENT,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });

  // Tier-colored wireframe materials (Section 4.4)
  const wireframe: Record<PartTier, THREE.MeshBasicMaterial> = {
    1: new THREE.MeshBasicMaterial({
      name: 'Wireframe_Tier1',
      color: 0x9cd8ff,
      wireframe: true,
      transparent: true,
      opacity: 0.72,
    }),
    2: new THREE.MeshBasicMaterial({
      name: 'Wireframe_Tier2',
      color: 0x3b9eff,
      wireframe: true,
      transparent: true,
      opacity: 0.85,
    }),
    3: new THREE.MeshBasicMaterial({
      name: 'Wireframe_Tier3',
      color: 0xf29f54,
      wireframe: true,
      transparent: true,
      opacity: 0.92,
    }),
  };

  const darkMask = new THREE.MeshBasicMaterial({
    name: 'SelectiveBloom_DarkMask',
    color: 0x000000,
  });

  const grades: Record<MaterialGrade, THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial> = {
    A: gradeA,
    B: gradeB,
    C: gradeC,
    D: gradeD,
    E: gradeE,
    F: gradeF,
  };

  const getBaseMaterial = (gradeKey: string): THREE.Material => {
    if (gradeKey === 'copper') return gradeFCopper;
    if (gradeKey === 'chrome') return gradeFChrome;
    if (gradeKey in grades) return grades[gradeKey as MaterialGrade];
    return gradeA;
  };

  const getHighlightMaterial = (gradeKey: string): THREE.Material => {
    if (gradeKey in highlights) return highlights[gradeKey as MaterialGradeKey];
    return highlights.A;
  };

  const setQuality = (quality: QualityLevel): void => {
    const isHigh = quality === 'High';
    gradeA.clearcoat = isHigh ? 0.60 : 0.0;
    gradeB.clearcoat = isHigh ? 0.25 : 0.0;
    gradeC.clearcoat = isHigh ? 0.10 : 0.0;
    gradeD.clearcoat = isHigh ? 1.00 : 0.20;
    gradeA.needsUpdate = true;
    gradeB.needsUpdate = true;
    gradeC.needsUpdate = true;
    gradeD.needsUpdate = true;
  };

  const dispose = (): void => {
    for (const m of Object.values(tierPlaceholder)) m.dispose();
    for (const m of Object.values(grades)) m.dispose();
    for (const [k, m] of Object.entries(highlights)) {
      if (k !== 'E') m.dispose();
    }
    for (const m of Object.values(wireframe)) m.dispose();
    xrayShell.dispose();
    gradeFCopper.dispose();
    gradeFChrome.dispose();
    darkMask.dispose();
    textures.armorBump?.dispose();
    textures.armorRoughness?.dispose();
    textures.metalRoughness?.dispose();
  };

  return {
    tierPlaceholder,
    grades,
    gradeFCopper,
    gradeFChrome,
    highlights,
    xrayShell,
    wireframe,
    darkMask,
    textures,
    getBaseMaterial,
    getHighlightMaterial,
    setQuality,
    dispose,
  };
}
