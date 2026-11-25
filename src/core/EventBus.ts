import type {
  CameraPreset,
  ExplodePreset,
  MotionMode,
  QualityLevel,
  RenderMode,
} from './AppState.ts';
import type { SemanticAxis, Side } from '../config/rig.ts';

export interface ManualAnglePayload {
  jointId: string;
  axis: SemanticAxis;
  valueDeg: number;
}

export interface ManualFingerPayload {
  side: Side;
  curlDeg: number;
}

export interface EventMap {
  'motion:change': MotionMode;
  'playback:toggle': boolean;
  'speed:change': number;
  'explode:slider': number;
  'explode:preset': Exclude<ExplodePreset, 'custom'>;
  'guidelines:toggle': boolean;
  'turntable:toggle': boolean;
  'bloom:toggle': boolean;
  'camera:preset': CameraPreset;
  'camera:ortho': boolean;
  'camera:reset': void;
  'reference:toggle': boolean;
  'reference:opacity': number;
  'renderMode:change': RenderMode;
  'quality:change': QualityLevel;
  'hud:toggle': boolean;
  'part:select': string | null;
  'part:hide': string;
  'part:isolate': string;
  'part:restoreAll': void;
  'manual:angle': ManualAnglePayload;
  'manual:finger': ManualFingerPayload;
  'manual:reset': void;
  'screenshot:export': void;
  'state:updated': void;
}

type Listener<T> = (payload: T) => void;

export class EventBus {
  private readonly listeners = new Map<keyof EventMap, Set<Listener<unknown>>>();

  on<K extends keyof EventMap>(event: K, listener: Listener<EventMap[K]>): () => void {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    set.add(listener as Listener<unknown>);
    return () => {
      set?.delete(listener as Listener<unknown>);
    };
  }

  emit<K extends keyof EventMap>(event: K, payload: EventMap[K]): void {
    const set = this.listeners.get(event);
    if (!set) return;
    for (const listener of set) {
      (listener as Listener<EventMap[K]>)(payload);
    }
  }
}
