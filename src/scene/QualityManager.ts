import type { MaterialBank } from '../config/materials.ts';
import type { QualityLevel } from '../core/AppState.ts';
import type { SelectiveBloomPipeline } from './PostProcessing.ts';
import type { StageManager } from './StageManager.ts';

const DOWNGRADE_FPS_THRESHOLD = 45;
const DOWNGRADE_DURATION_SEC = 3.0;
const WARMUP_GRACE_SEC = 2.5;

/**
 * Manages High / Low quality presets and automatic FPS downgrade (Section 7.3):
 * If average FPS remains < 45 for 3 continuous seconds on High, switches to Low and notifies UI.
 * Users can manually switch back to High at any time (which suppresses repeated auto-downgrades).
 */
export class QualityManager {
  private quality: QualityLevel;
  private lowFpsTimer = 0;
  private warmupTimer = 0;
  private manualOverride = false;

  constructor(
    initialQuality: QualityLevel,
    private readonly stage: StageManager,
    private readonly materials: MaterialBank,
    private readonly postFX: SelectiveBloomPipeline,
    private readonly onAutoDowngrade: (nextQuality: QualityLevel) => void,
  ) {
    this.quality = initialQuality;
  }

  public getQuality(): QualityLevel {
    return this.quality;
  }

  public setQuality(
    quality: QualityLevel,
    width: number,
    height: number,
    isManual = true,
  ): void {
    this.quality = quality;
    if (isManual) {
      this.manualOverride = quality === 'High';
      this.lowFpsTimer = 0;
    }
    this.stage.setQuality(quality);
    this.materials.setQuality(quality);
    this.postFX.setQuality(quality, width, height);
  }

  public sampleFps(fps: number, dtWindow: number, width: number, height: number): void {
    if (this.quality !== 'High' || this.manualOverride) {
      this.lowFpsTimer = 0;
      return;
    }

    this.warmupTimer += dtWindow;
    if (this.warmupTimer < WARMUP_GRACE_SEC) return;

    if (fps < DOWNGRADE_FPS_THRESHOLD) {
      this.lowFpsTimer += dtWindow;
      if (this.lowFpsTimer >= DOWNGRADE_DURATION_SEC) {
        this.lowFpsTimer = 0;
        this.setQuality('Low', width, height, false);
        this.onAutoDowngrade('Low');
      }
    } else {
      this.lowFpsTimer = Math.max(0, this.lowFpsTimer - dtWindow * 0.5);
    }
  }
}
