import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { easeInOutCubic, type CameraPreset } from '../core/AppState.ts';
import type { AssembledPart } from '../robot/RobotAssembler.ts';

const PRESET_TRANSITION_DURATION = 0.8;
const FOCUS_TRANSITION_DURATION = 0.6;
const ORTHO_VIEW_HEIGHT = 2.34; // Frames full 1.80m robot + feet cleanly between top & bottom HUD bars

interface PresetTransform {
  pos: readonly [number, number, number];
  target: readonly [number, number, number];
}

const CAMERA_PRESETS: Record<Exclude<CameraPreset, 'FREE'>, PresetTransform> = {
  FRONT: { pos: [0, 0.90, 3.6], target: [0, 0.90, 0] },
  // Side view from robot's right (-X) so robot (+Z forward) faces screen right (matches reference image)
  SIDE: { pos: [-3.6, 0.90, 0], target: [0, 0.90, 0] },
  BACK: { pos: [0, 0.90, -3.6], target: [0, 0.90, 0] },
  '3/4': { pos: [-2.55, 1.48, 3.55], target: [0, 0.99, 0] },
};

// Module-level scratch objects (zero per-frame heap allocations)
const _partBox = new THREE.Box3();
const _partSphere = new THREE.Sphere();
const _followCenter = new THREE.Vector3();
const _followDelta = new THREE.Vector3();
const _viewDir = new THREE.Vector3();

export class CameraController {
  public readonly perspCamera: THREE.PerspectiveCamera;
  public readonly orthoCamera: THREE.OrthographicCamera;
  public activeCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
  public readonly controls: OrbitControls;

  private isOrtho = false;
  private aspect = 1;
  private transitionDuration = PRESET_TRANSITION_DURATION;
  private transitionElapsed = PRESET_TRANSITION_DURATION;
  private readonly startPos = new THREE.Vector3();
  private readonly endPos = new THREE.Vector3();
  private readonly startTarget = new THREE.Vector3();
  private readonly endTarget = new THREE.Vector3();
  private readonly onControlsStart: () => void;

  // Live part tracking when inspecting a clicked part during animation (Section 4.3.3)
  private trackedPart: AssembledPart | null = null;
  private readonly lastTrackedCenter = new THREE.Vector3();
  private hasTrackedCenter = false;

  constructor(
    domElement: HTMLElement,
    width: number,
    height: number,
    onUserOrbit?: () => void,
  ) {
    this.aspect = Math.max(0.1, width / Math.max(1, height));

    this.perspCamera = new THREE.PerspectiveCamera(38, this.aspect, 0.05, 60);
    const halfH = ORTHO_VIEW_HEIGHT / 2;
    const halfW = halfH * this.aspect;
    this.orthoCamera = new THREE.OrthographicCamera(-halfW, halfW, halfH, -halfH, 0.05, 60);

    this.activeCamera = this.perspCamera;
    this.controls = new OrbitControls(this.activeCamera, domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 10.0;
    // Allow exact horizontal (90 deg) FRONT/SIDE/BACK views without clamping down to 89.1 deg
    this.controls.maxPolarAngle = Math.PI * 0.501;

    this.onControlsStart = (): void => {
      // Immediately yield camera control when user drags/zooms instead of fighting active lerp
      this.transitionElapsed = this.transitionDuration;
      onUserOrbit?.();
    };
    this.controls.addEventListener('start', this.onControlsStart);

    this.setPreset('3/4', true);
  }

  public setPreset(preset: CameraPreset, immediate = false): void {
    if (preset === 'FREE') return;
    this.trackedPart = null;
    this.hasTrackedCenter = false;

    if (this.isOrtho) {
      this.orthoCamera.zoom = 1.0;
      this.orthoCamera.updateProjectionMatrix();
    }

    const cfg = CAMERA_PRESETS[preset];
    this.startPos.copy(this.activeCamera.position);
    this.startTarget.copy(this.controls.target);
    this.endPos.set(cfg.pos[0], cfg.pos[1], cfg.pos[2]);
    this.endTarget.set(cfg.target[0], cfg.target[1], cfg.target[2]);
    this.transitionDuration = PRESET_TRANSITION_DURATION;

    if (immediate) {
      this.transitionElapsed = this.transitionDuration;
      this.activeCamera.position.copy(this.endPos);
      this.controls.target.copy(this.endTarget);
      this.syncInactiveCamera();
      this.controls.update();
    } else {
      this.transitionElapsed = 0;
    }
  }

  /**
   * Smoothly focuses camera over 0.6s onto a part's world bounding sphere
   * and continuously follows the part center while animation plays.
   */
  public focusPart(part: AssembledPart): void {
    // Switch out of ortho if active so focus framing works cleanly in 3D
    if (this.isOrtho) {
      this.setOrthographic(false);
    }

    _partBox.setFromObject(part.container);
    if (_partBox.isEmpty()) return;
    _partBox.getBoundingSphere(_partSphere);

    this.startPos.copy(this.activeCamera.position);
    this.startTarget.copy(this.controls.target);
    this.endTarget.copy(_partSphere.center);

    // Keep current viewing direction, pull to comfortable inspection distance
    _viewDir.copy(this.activeCamera.position).sub(this.controls.target);
    if (_viewDir.lengthSq() < 1e-4) {
      _viewDir.set(-0.5, 0.2, 0.85);
    }
    _viewDir.normalize();

    const dist = Math.max(0.62, Math.min(2.2, _partSphere.radius * 4.2));
    this.endPos.copy(_partSphere.center).addScaledVector(_viewDir, dist);
    this.endPos.y = Math.max(0.12, this.endPos.y);

    this.transitionDuration = FOCUS_TRANSITION_DURATION;
    this.transitionElapsed = 0;
    this.trackedPart = part;
    this.lastTrackedCenter.copy(_partSphere.center);
    this.hasTrackedCenter = true;
  }

  public clearFocus(): void {
    this.trackedPart = null;
    this.hasTrackedCenter = false;
  }

  public setOrthographic(enabled: boolean): void {
    if (this.isOrtho === enabled) return;
    const prevCam = this.activeCamera;
    this.isOrtho = enabled;
    this.activeCamera = enabled ? this.orthoCamera : this.perspCamera;

    this.activeCamera.position.copy(prevCam.position);
    this.activeCamera.quaternion.copy(prevCam.quaternion);
    if (enabled) {
      this.orthoCamera.zoom = 1.0;
      this.orthoCamera.updateProjectionMatrix();
    }
    this.controls.object = this.activeCamera;
    this.controls.update();
  }

  public isOrthographic(): boolean {
    return this.isOrtho;
  }

  public resize(width: number, height: number): void {
    this.aspect = Math.max(0.1, width / Math.max(1, height));
    this.perspCamera.aspect = this.aspect;
    this.perspCamera.updateProjectionMatrix();

    const halfH = ORTHO_VIEW_HEIGHT / 2;
    const halfW = halfH * this.aspect;
    this.orthoCamera.left = -halfW;
    this.orthoCamera.right = halfW;
    this.orthoCamera.top = halfH;
    this.orthoCamera.bottom = -halfH;
    this.orthoCamera.updateProjectionMatrix();
  }

  public update(dt: number): void {
    // Follow moving inspected part center during animation
    if (this.trackedPart && this.trackedPart.container.visible) {
      _followCenter
        .copy(this.trackedPart.localCenter)
        .applyMatrix4(this.trackedPart.container.matrixWorld);

      if (this.hasTrackedCenter) {
        _followDelta.subVectors(_followCenter, this.lastTrackedCenter);
        if (this.transitionElapsed < this.transitionDuration) {
          this.endTarget.add(_followDelta);
          this.endPos.add(_followDelta);
        } else {
          this.controls.target.add(_followDelta);
          this.activeCamera.position.add(_followDelta);
        }
      }
      this.lastTrackedCenter.copy(_followCenter);
      this.hasTrackedCenter = true;
    }

    if (this.transitionElapsed < this.transitionDuration) {
      this.transitionElapsed = Math.min(this.transitionDuration, this.transitionElapsed + dt);
      const alpha = easeInOutCubic(this.transitionElapsed / this.transitionDuration);
      this.activeCamera.position.lerpVectors(this.startPos, this.endPos, alpha);
      this.controls.target.lerpVectors(this.startTarget, this.endTarget, alpha);
      this.syncInactiveCamera();
    }

    this.controls.update();
  }

  private syncInactiveCamera(): void {
    const other = this.isOrtho ? this.perspCamera : this.orthoCamera;
    other.position.copy(this.activeCamera.position);
    other.quaternion.copy(this.activeCamera.quaternion);
  }

  public dispose(): void {
    this.controls.removeEventListener('start', this.onControlsStart);
    this.controls.dispose();
  }
}
