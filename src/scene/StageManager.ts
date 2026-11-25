import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { Reflector } from 'three/examples/jsm/objects/Reflector.js';
import type { QualityLevel } from '../core/AppState.ts';
import { BLOOM_LAYER } from '../robot/RobotAssembler.ts';

function createRadialBackgroundTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;
  const grad = ctx.createRadialGradient(256, 240, 20, 256, 256, 360);
  grad.addColorStop(0, '#101A2A');
  grad.addColorStop(0.55, '#09101C');
  grad.addColorStop(1, '#05080D');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createHoloRingTexture(): THREE.CanvasTexture {
  const size = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const cx = size / 2;
  const cy = size / 2;

  ctx.clearRect(0, 0, size, size);

  const glow = ctx.createRadialGradient(cx, cy, 40, cx, cy, size * 0.48);
  glow.addColorStop(0, 'rgba(31, 134, 255, 0.14)');
  glow.addColorStop(0.55, 'rgba(31, 134, 255, 0.05)');
  glow.addColorStop(1, 'rgba(31, 134, 255, 0.0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, size, size);

  const rings = [
    { r: 445, w: 3.0, alpha: 0.72 },
    { r: 418, w: 1.5, alpha: 0.35 },
    { r: 345, w: 2.2, alpha: 0.55 },
    { r: 245, w: 1.4, alpha: 0.32 },
    { r: 155, w: 1.2, alpha: 0.25 },
  ];
  for (const ring of rings) {
    ctx.beginPath();
    ctx.arc(cx, cy, ring.r, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(78, 165, 242, ${ring.alpha})`;
    ctx.lineWidth = ring.w;
    ctx.stroke();
  }

  for (let i = 0; i < 72; i++) {
    const angle = (i / 72) * Math.PI * 2;
    const isMajor = i % 6 === 0;
    const r0 = isMajor ? 422 : 434;
    const r1 = 445;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * r0, cy + Math.sin(angle) * r0);
    ctx.lineTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
    ctx.strokeStyle = isMajor ? 'rgba(120, 200, 255, 0.8)' : 'rgba(78, 165, 242, 0.35)';
    ctx.lineWidth = isMajor ? 2.2 : 1.1;
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(78, 165, 242, 0.22)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx - 460, cy);
  ctx.lineTo(cx + 460, cy);
  ctx.moveTo(cx, cy - 460);
  ctx.lineTo(cx, cy + 460);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function createScanArcTexture(): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const cx = size / 2;
  const cy = size / 2;

  ctx.clearRect(0, 0, size, size);
  ctx.lineWidth = 5;
  ctx.strokeStyle = 'rgba(88, 195, 255, 0.85)';

  ctx.beginPath();
  ctx.arc(cx, cy, 208, 0.15, 1.15);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 208, Math.PI + 0.15, Math.PI + 1.15);
  ctx.stroke();

  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(56, 165, 255, 0.55)';
  ctx.beginPath();
  ctx.arc(cx, cy, 168, 2.0, 2.8);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createTreadmillGridTexture(): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, size, size);

  ctx.strokeStyle = 'rgba(78, 165, 242, 0.35)';
  ctx.lineWidth = 2;
  const step = size / 8;
  for (let i = 0; i <= 8; i++) {
    const p = i * step;
    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(size, p);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, size);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 4);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class StageManager {
  public readonly group = new THREE.Group();
  public readonly reflector: Reflector;

  private readonly bgTex: THREE.CanvasTexture;
  private readonly holoTex: THREE.CanvasTexture;
  private readonly scanTex: THREE.CanvasTexture;
  private readonly gridTex: THREE.CanvasTexture;
  private readonly envRT: THREE.WebGLRenderTarget;

  private readonly reflectorOverlayMesh: THREE.Mesh;
  private readonly shadowCatcherMesh: THREE.Mesh;
  private readonly lowFloorMesh: THREE.Mesh;
  private readonly holoMesh: THREE.Mesh;
  private readonly scanMesh: THREE.Mesh;
  private readonly treadmillMesh: THREE.Mesh;
  private readonly keyLight: THREE.DirectionalLight;
  private readonly disposables: Array<{ dispose: () => void }> = [];

  constructor(
    scene: THREE.Scene,
    renderer: THREE.WebGLRenderer,
    width: number,
    height: number,
    initialQuality: QualityLevel = 'High',
  ) {
    this.group.name = 'studio_stage';
    scene.add(this.group);

    // Background & Fog (Section 6.1)
    this.bgTex = createRadialBackgroundTexture();
    scene.background = this.bgTex;
    scene.fog = new THREE.FogExp2(0x070b12, 0.04);

    // Procedural RoomEnvironment PMREM
    const pmrem = new THREE.PMREMGenerator(renderer);
    const roomEnv = new RoomEnvironment();
    this.envRT = pmrem.fromScene(roomEnv, 0.04);
    scene.environment = this.envRT.texture;
    scene.environmentIntensity = 0.42;
    roomEnv.dispose();
    pmrem.dispose();

    // --- High-tier layered reflective floor (Section 6.1) ---
    // Layer 0 (y = 0.000): 0.5x resolution Reflector
    const refGeo = new THREE.CircleGeometry(10, 48);
    this.reflector = new Reflector(refGeo, {
      clipBias: 0.003,
      textureWidth: Math.max(256, Math.floor(width * 0.5)),
      textureHeight: Math.max(256, Math.floor(height * 0.5)),
      color: 0x5c7291,
    });
    this.reflector.rotation.x = -Math.PI / 2;
    this.reflector.position.y = 0.0;
    this.group.add(this.reflector);
    this.disposables.push(refGeo);

    // Layer 1 (y = 0.001): Semi-transparent dark smoked overlay (softens & darkens mirror)
    const overlayGeo = new THREE.CircleGeometry(14, 64);
    overlayGeo.rotateX(-Math.PI / 2);
    const overlayMat = new THREE.MeshStandardMaterial({
      color: 0x050912,
      metalness: 0.25,
      roughness: 0.58,
      transparent: true,
      opacity: 0.66,
      depthWrite: false,
      envMapIntensity: 0.18,
    });
    this.reflectorOverlayMesh = new THREE.Mesh(overlayGeo, overlayMat);
    this.reflectorOverlayMesh.position.y = 0.001;
    this.group.add(this.reflectorOverlayMesh);
    this.disposables.push(overlayGeo, overlayMat);

    // Layer 2 (y = 0.002): ShadowMaterial catcher (Reflector itself doesn't receive shadows)
    const shadowGeo = new THREE.CircleGeometry(6, 48);
    shadowGeo.rotateX(-Math.PI / 2);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.52 });
    this.shadowCatcherMesh = new THREE.Mesh(shadowGeo, shadowMat);
    this.shadowCatcherMesh.position.y = 0.002;
    this.shadowCatcherMesh.receiveShadow = true;
    this.group.add(this.shadowCatcherMesh);
    this.disposables.push(shadowGeo, shadowMat);

    // Low-tier fallback floor (dark metallic plane receiving shadows, no Reflector pass)
    const lowFloorGeo = new THREE.CircleGeometry(14, 64);
    lowFloorGeo.rotateX(-Math.PI / 2);
    const lowFloorMat = new THREE.MeshStandardMaterial({
      color: 0x070d18,
      metalness: 0.30,
      roughness: 0.54,
      envMapIntensity: 0.24,
    });
    this.lowFloorMesh = new THREE.Mesh(lowFloorGeo, lowFloorMat);
    this.lowFloorMesh.position.y = 0.0;
    this.lowFloorMesh.receiveShadow = true;
    this.group.add(this.lowFloorMesh);
    this.disposables.push(lowFloorGeo, lowFloorMat);

    // Layer 3 (y = 0.003): Holographic calibration ring
    this.holoTex = createHoloRingTexture();
    const holoGeo = new THREE.PlaneGeometry(2.2, 2.2);
    holoGeo.rotateX(-Math.PI / 2);
    const holoMat = new THREE.MeshBasicMaterial({
      map: this.holoTex,
      transparent: true,
      opacity: 0.82,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.holoMesh = new THREE.Mesh(holoGeo, holoMat);
    this.holoMesh.position.y = 0.003;
    this.group.add(this.holoMesh);
    this.disposables.push(holoGeo, holoMat);

    // Counter-rotating scan arc (subtle selective bloom layer)
    this.scanTex = createScanArcTexture();
    const scanGeo = new THREE.PlaneGeometry(2.2, 2.2);
    scanGeo.rotateX(-Math.PI / 2);
    const scanMat = new THREE.MeshBasicMaterial({
      map: this.scanTex,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.scanMesh = new THREE.Mesh(scanGeo, scanMat);
    this.scanMesh.position.y = 0.0035;
    this.scanMesh.layers.enable(BLOOM_LAYER);
    this.group.add(this.scanMesh);
    this.disposables.push(scanGeo, scanMat);

    // Treadmill scrolling grid (visible during Walk / Run)
    this.gridTex = createTreadmillGridTexture();
    const gridGeo = new THREE.PlaneGeometry(2.4, 2.4);
    gridGeo.rotateX(-Math.PI / 2);
    const gridMat = new THREE.MeshBasicMaterial({
      map: this.gridTex,
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.treadmillMesh = new THREE.Mesh(gridGeo, gridMat);
    this.treadmillMesh.position.y = 0.0025;
    this.group.add(this.treadmillMesh);
    this.disposables.push(gridGeo, gridMat);

    // Skip reflecting floor-plane overlays sitting right on y=0.001..0.0035 beneath themselves
    const origReflectorBeforeRender = this.reflector.onBeforeRender;
    this.reflector.onBeforeRender = (rendererArg, sceneArg, camArg, geoArg, matArg, grpArg) => {
      const ov = this.reflectorOverlayMesh.visible;
      const sh = this.shadowCatcherMesh.visible;
      const ho = this.holoMesh.visible;
      const sc = this.scanMesh.visible;
      const tm = this.treadmillMesh.visible;
      this.reflectorOverlayMesh.visible = false;
      this.shadowCatcherMesh.visible = false;
      this.holoMesh.visible = false;
      this.scanMesh.visible = false;
      this.treadmillMesh.visible = false;
      origReflectorBeforeRender.call(
        this.reflector,
        rendererArg,
        sceneArg,
        camArg,
        geoArg,
        matArg,
        grpArg,
      );
      this.reflectorOverlayMesh.visible = ov;
      this.shadowCatcherMesh.visible = sh;
      this.holoMesh.visible = ho;
      this.scanMesh.visible = sc;
      this.treadmillMesh.visible = tm;
    };

    // Four-point studio lighting (Section 6.2)
    this.keyLight = new THREE.DirectionalLight(0xf0f6ff, 1.85);
    this.keyLight.position.set(3.5, 5.5, 4.0);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.camera.near = 0.5;
    this.keyLight.shadow.camera.far = 15;
    this.keyLight.shadow.camera.left = -2.2;
    this.keyLight.shadow.camera.right = 2.2;
    this.keyLight.shadow.camera.top = 2.6;
    this.keyLight.shadow.camera.bottom = -1.2;
    this.keyLight.shadow.bias = -0.0004;
    this.group.add(this.keyLight);

    const fillLight = new THREE.DirectionalLight(0x6b9bd2, 0.65);
    fillLight.position.set(-4.0, 2.5, 2.5);
    this.group.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x3a8bff, 2.1);
    rimLight.position.set(0.0, 4.0, -4.5);
    this.group.add(rimLight);

    const floorBounce = new THREE.HemisphereLight(0x4ea5f2, 0x0b1a30, 0.35);
    this.group.add(floorBounce);

    // Subtle blue ground-ring up-light
    const ringGlowLight = new THREE.PointLight(0x288eff, 0.35, 3.2, 2);
    ringGlowLight.position.set(0, 0.18, 0);
    this.group.add(ringGlowLight);

    this.setQuality(initialQuality);
  }

  public setQuality(quality: QualityLevel): void {
    const isHigh = quality === 'High';
    this.reflector.visible = isHigh;
    this.reflectorOverlayMesh.visible = isHigh;
    this.shadowCatcherMesh.visible = isHigh;
    this.lowFloorMesh.visible = !isHigh;

    const shadowSize = isHigh ? 2048 : 1024;
    if (this.keyLight.shadow.mapSize.width !== shadowSize) {
      this.keyLight.shadow.mapSize.set(shadowSize, shadowSize);
      this.keyLight.shadow.map?.dispose();
      this.keyLight.shadow.map = null;
    }
  }

  public resize(width: number, height: number): void {
    this.reflector
      .getRenderTarget()
      .setSize(Math.max(256, Math.floor(width * 0.5)), Math.max(256, Math.floor(height * 0.5)));
  }

  public update(dt: number, treadmillSpeed: number): void {
    this.holoMesh.rotation.y += dt * 0.08;
    this.scanMesh.rotation.y -= dt * 0.28;

    const gridMat = this.treadmillMesh.material as THREE.MeshBasicMaterial;
    if (treadmillSpeed > 0.01) {
      gridMat.opacity = Math.min(0.45, gridMat.opacity + dt * 2.5);
      this.gridTex.offset.y -= (treadmillSpeed * dt) / 0.6;
    } else {
      gridMat.opacity = Math.max(0, gridMat.opacity - dt * 2.5);
    }
  }

  public dispose(): void {
    this.reflector.dispose();
    this.bgTex.dispose();
    this.holoTex.dispose();
    this.scanTex.dispose();
    this.gridTex.dispose();
    this.envRT.dispose();
    for (const d of this.disposables) d.dispose();
  }
}
