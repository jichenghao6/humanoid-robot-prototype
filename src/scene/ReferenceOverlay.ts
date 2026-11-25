import * as THREE from 'three';
import { ROBOT_TOTAL_HEIGHT } from '../config/proportions.ts';
import type { CameraPreset } from '../core/AppState.ts';

export interface ReferenceCalibration {
  imgWidth: number;
  imgHeight: number;
  crownY: number;
  soleY: number;
  frontX0: number;
  frontX1: number;
  sideX0: number;
  sideX1: number;
  backX0: number;
  backX1: number;
  offsetXFront: number;
  offsetXSide: number;
  offsetXBack: number;
  offsetY: number;
}

/**
 * Orthographic Three-View Reference Overlay (Section 6.4):
 * Displays the exact cropped region of public/reference/three-view.png (1536x1024)
 * behind the robot in orthographic FRONT / SIDE / BACK views, calibrated so crown->sole = 1.800 m.
 * Automatically disables itself with zero errors if the image does not exist.
 */
export class ReferenceOverlay {
  public readonly group = new THREE.Group();
  public available = false;

  public readonly calibration: ReferenceCalibration = {
    imgWidth: 1536,
    imgHeight: 1024,
    crownY: 72,
    soleY: 923,
    frontX0: 168,
    frontX1: 543,
    sideX0: 660,
    sideX1: 920,
    backX0: 1044,
    backX1: 1423,
    offsetXFront: 0,
    offsetXSide: 0.022,
    offsetXBack: 0,
    offsetY: 0,
  };

  private texture: THREE.Texture | null = null;
  private readonly planeGeo: THREE.PlaneGeometry;
  private readonly planeMat: THREE.MeshBasicMaterial;
  private readonly planeMesh: THREE.Mesh;

  private enabled = false;
  private opacity = 0.45;
  private activeView: CameraPreset = '3/4';
  private isOrtho = false;

  constructor(
    scene: THREE.Scene,
    onAvailabilityChange?: (available: boolean) => void,
  ) {
    this.group.name = 'reference_overlay';
    this.group.visible = false;
    scene.add(this.group);

    this.planeGeo = new THREE.PlaneGeometry(1, 1);
    this.planeMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: this.opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
      fog: false,
    });
    this.planeMesh = new THREE.Mesh(this.planeGeo, this.planeMat);
    this.group.add(this.planeMesh);

    const loader = new THREE.TextureLoader();
    loader.load(
      '/reference/three-view.png',
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        const img = tex.image as { width?: number; height?: number } | undefined;
        if (img?.width && img?.height) {
          this.calibration.imgWidth = img.width;
          this.calibration.imgHeight = img.height;
        }
        this.texture = tex;
        this.planeMat.map = tex;
        this.planeMat.needsUpdate = true;
        this.available = true;
        this.updateTransform();
        onAvailabilityChange?.(true);
      },
      undefined,
      () => {
        // Gracefully disable if reference image is missing (Section 1.3 & 6.4)
        this.available = false;
        this.group.visible = false;
        onAvailabilityChange?.(false);
      },
    );
  }

  public setState(enabled: boolean, opacity: number, view: CameraPreset, isOrtho: boolean): void {
    this.enabled = enabled;
    this.opacity = opacity;
    this.activeView = view;
    this.isOrtho = isOrtho;
    this.planeMat.opacity = opacity;
    this.updateTransform();
  }

  public updateTransform(): void {
    const validView =
      this.activeView === 'FRONT' || this.activeView === 'SIDE' || this.activeView === 'BACK';

    if (!this.available || !this.texture || !this.enabled || !this.isOrtho || !validView) {
      this.group.visible = false;
      return;
    }

    const c = this.calibration;
    const w = Math.max(1, c.imgWidth);
    const h = Math.max(1, c.imgHeight);

    let x0 = c.frontX0;
    let x1 = c.frontX1;
    let offsetX = c.offsetXFront;
    let viewCrownY = c.crownY - 4;
    let viewSoleY = c.soleY - 1;
    if (this.activeView === 'SIDE') {
      x0 = c.sideX0;
      x1 = c.sideX1;
      offsetX = c.offsetXSide;
      viewCrownY = c.crownY + 4;
      viewSoleY = c.soleY + 1;
    } else if (this.activeView === 'BACK') {
      x0 = c.backX0;
      x1 = c.backX1;
      offsetX = c.offsetXBack;
      viewCrownY = c.crownY + 1;
      viewSoleY = c.soleY + 1;
    }

    // Vertical scale: pixel span (viewSoleY - viewCrownY) maps to exactly ROBOT_TOTAL_HEIGHT (1.800 m)
    const pixelHeightRobot = Math.max(10, viewSoleY - viewCrownY);
    const metersPerPixel = ROBOT_TOTAL_HEIGHT / pixelHeightRobot;

    // Tight vertical crop excluding header text labels ("HUMANOID ROBOT", "SIDE VIEW", "BACK VIEW")
    const cropYTop = Math.max(0, viewCrownY - 4);
    const cropYBot = Math.min(h, viewSoleY + 12);
    const cropPixW = Math.max(10, x1 - x0);
    const cropPixH = Math.max(10, cropYBot - cropYTop);

    const worldW = cropPixW * metersPerPixel;
    const worldH = cropPixH * metersPerPixel;

    // World Y of crop center: viewSoleY is at Y=0, viewCrownY is at Y=1.800
    const cropCenterPixY = (cropYTop + cropYBot) * 0.5;
    const centerWorldY = (viewSoleY - cropCenterPixY) * metersPerPixel + c.offsetY;

    // Texture UV mapping (OpenGL UV v=0 is image bottom, v=1 is image top)
    const u0 = x0 / w;
    const uSpan = cropPixW / w;
    const vBottom = 1 - cropYBot / h;
    const vSpan = cropPixH / h;

    this.texture.offset.set(u0, vBottom);
    this.texture.repeat.set(uSpan, vSpan);
    this.texture.needsUpdate = true;

    this.planeMesh.scale.set(worldW, worldH, 1);

    // Position plane strictly behind the robot along each orthographic view axis
    if (this.activeView === 'FRONT') {
      // Camera at +Z looking toward -Z -> plane behind robot at z = -1.2
      this.planeMesh.position.set(offsetX, centerWorldY, -1.2);
      this.planeMesh.rotation.set(0, 0, 0);
    } else if (this.activeView === 'SIDE') {
      // Camera at -X looking toward +X (screen right = +Z) -> plane behind robot at x = +1.2
      this.planeMesh.position.set(1.2, centerWorldY, offsetX);
      this.planeMesh.rotation.set(0, -Math.PI / 2, 0);
    } else {
      // BACK: Camera at -Z looking toward +Z -> plane behind robot at z = +1.2
      this.planeMesh.position.set(offsetX, centerWorldY, 1.2);
      this.planeMesh.rotation.set(0, Math.PI, 0);
    }

    this.group.visible = true;
  }

  public dispose(): void {
    this.texture?.dispose();
    this.planeGeo.dispose();
    this.planeMat.dispose();
  }
}
