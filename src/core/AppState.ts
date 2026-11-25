export type MotionMode = 'Idle' | 'Walk' | 'Run' | 'Jump' | 'Wave' | 'Manual';

export type RenderMode = 'PBR' | 'X-Ray' | 'Wireframe';

export type QualityLevel = 'High' | 'Low';

export type CameraPreset = 'FRONT' | 'SIDE' | 'BACK' | '3/4' | 'FREE';

export type ExplodePreset = 'assembled' | 'shell_off' | 'full_explode' | 'custom';

export interface ExplodeState {
  t1: number;
  t2: number;
  t3: number;
}

export interface AppStateData {
  motion: MotionMode;
  playing: boolean;
  speed: number;
  explodeSlider: number;
  explodeTarget: ExplodeState;
  explodeCurrent: ExplodeState;
  explodePreset: ExplodePreset;
  showGuideLines: boolean;
  turntable: boolean;
  renderMode: RenderMode;
  cameraPreset: CameraPreset;
  orthographic: boolean;
  referenceOverlay: boolean;
  referenceOpacity: number;
  selectedPartId: string | null;
  hoveredPartId: string | null;
  hiddenPartIds: Set<string>;
  quality: QualityLevel;
  hudVisible: boolean;
  fps: number;
  drawCalls: number;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function easeInOutCubic(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  return c < 0.5 ? 4 * c * c * c : 1 - Math.pow(-2 * c + 2, 3) / 2;
}

export function sliderToExplodeTargets(s: number, out: ExplodeState): ExplodeState {
  const clamped = Math.max(0, Math.min(1, s));
  out.t1 = smoothstep(0.0, 0.45, clamped);
  out.t2 = smoothstep(0.35, 0.80, clamped);
  out.t3 = smoothstep(0.65, 1.00, clamped);
  return out;
}

export const EXPLODE_PRESETS: Record<Exclude<ExplodePreset, 'custom'>, ExplodeState> = {
  assembled: { t1: 0, t2: 0, t3: 0 },
  shell_off: { t1: 0.6, t2: 0, t3: 0 },
  full_explode: { t1: 1, t2: 1, t3: 1 },
};

export function createInitialAppState(): AppStateData {
  return {
    motion: 'Idle',
    playing: true,
    speed: 1.0,
    explodeSlider: 0,
    explodeTarget: { t1: 0, t2: 0, t3: 0 },
    explodeCurrent: { t1: 0, t2: 0, t3: 0 },
    explodePreset: 'assembled',
    showGuideLines: true,
    turntable: false,
    renderMode: 'PBR',
    cameraPreset: '3/4',
    orthographic: false,
    referenceOverlay: false,
    referenceOpacity: 0.45,
    selectedPartId: null,
    hoveredPartId: null,
    hiddenPartIds: new Set<string>(),
    quality: 'High',
    hudVisible: true,
    fps: 60,
    drawCalls: 0,
  };
}
