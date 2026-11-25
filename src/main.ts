import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/600.css';
import '@fontsource/jetbrains-mono/700.css';
import GUI from 'lil-gui';
import * as THREE from 'three';
import { createMaterialBank } from './config/materials.ts';
import {
  EXPLODE_PRESETS,
  createInitialAppState,
  sliderToExplodeTargets,
  type CameraPreset,
  type ExplodePreset,
  type MotionMode,
  type QualityLevel,
  type RenderMode,
} from './core/AppState.ts';
import { EventBus } from './core/EventBus.ts';
import { AnimationEngine } from './robot/AnimationEngine.ts';
import { ExplodeController } from './robot/ExplodeController.ts';
import { InteractionController } from './robot/InteractionController.ts';
import { PartRegistry } from './robot/PartRegistry.ts';
import { RenderModeController } from './robot/RenderModeController.ts';
import { RobotAssembler } from './robot/RobotAssembler.ts';
import { RobotRig } from './robot/RobotRig.ts';
import { registerArmBuilders } from './robot/parts/ArmBuilder.ts';
import { registerHeadBuilders } from './robot/parts/HeadBuilder.ts';
import { registerLegBuilders } from './robot/parts/LegBuilder.ts';
import { registerTorsoBuilders } from './robot/parts/TorsoBuilder.ts';
import { CameraController } from './scene/CameraController.ts';
import { SelectiveBloomPipeline } from './scene/PostProcessing.ts';
import { QualityManager } from './scene/QualityManager.ts';
import { ReferenceOverlay } from './scene/ReferenceOverlay.ts';
import { StageManager } from './scene/StageManager.ts';
import { HudController } from './ui/HudController.ts';
import './ui/hud.css';

interface PrototypeDebugAPI {
  setCamera: (preset: CameraPreset, ortho?: boolean) => void;
  setMotion: (mode: MotionMode, phase?: number) => void;
  setExplode: (s: number | Exclude<ExplodePreset, 'custom'>, snap?: boolean) => void;
  setRenderMode: (mode: RenderMode) => void;
  setQuality: (quality: QualityLevel) => void;
  setReferenceOverlay: (enabled: boolean, opacity?: number) => void;
  selectPart: (partId: string | null) => void;
  setBloom: (enabled: boolean) => void;
  setHudVisible: (visible: boolean) => void;
  setGlbOverrides: (enabled: boolean) => Promise<string[]>;
  getDrawCalls: () => number;
  getMemory: () => { geometries: number; textures: number };
}

declare global {
  interface Window {
    __PROTOTYPE_DEBUG__?: PrototypeDebugAPI;
  }
}

function bootstrap(): void {
  const canvas = document.getElementById('viewport') as HTMLCanvasElement;
  const hudRoot = document.getElementById('hud-root') as HTMLElement;

  const state = createInitialAppState();
  const bus = new EventBus();

  // Renderer
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  // Scene & Layered High/Low Stage
  const scene = new THREE.Scene();
  const stage = new StageManager(
    scene,
    renderer,
    window.innerWidth,
    window.innerHeight,
    state.quality,
  );
  const cameraCtrl = new CameraController(
    canvas,
    window.innerWidth,
    window.innerHeight,
    () => {
      if (state.cameraPreset !== 'FREE') {
        state.cameraPreset = 'FREE';
        hud.sync();
      }
    },
  );

  // Materials (6-grade PBR + highlight + X-Ray Fresnel + Wireframe variants)
  const materials = createMaterialBank(state.quality);

  // Full-body robot assembly (ALL 57 BOM parts detailed)
  const rig = new RobotRig();
  const registry = new PartRegistry();
  registerHeadBuilders(registry);
  registerTorsoBuilders(registry);
  registerArmBuilders(registry, ['L', 'R']);
  registerLegBuilders(registry, ['L', 'R']);

  const assembler = new RobotAssembler();
  assembler.assemble(rig, registry, materials);
  scene.add(rig.root);

  const explodeCtrl = new ExplodeController(assembler.partsList);
  scene.add(explodeCtrl.guideGroup);

  const renderModeCtrl = new RenderModeController(assembler.partsList, materials);

  // HUD
  const hud = new HudController(hudRoot, state, bus, rig);

  // Orthographic Three-View Reference Overlay (Section 6.4)
  const refOverlay = new ReferenceOverlay(scene, (available) => {
    hud.setReferenceAvailable(available);
    hud.sync();
  });

  // Layer-based Selective Bloom pipeline (Section 6.3)
  const postFX = new SelectiveBloomPipeline(
    renderer,
    scene,
    cameraCtrl.activeCamera,
    materials.darkMask,
    window.innerWidth,
    window.innerHeight,
  );
  postFX.excludeFromBloomPass(explodeCtrl.guideGroup);
  postFX.excludeFromBloomPass(refOverlay.group);
  postFX.excludeFromBloomPass(stage.reflector);
  postFX.setQuality(state.quality, window.innerWidth, window.innerHeight);

  // Quality Manager with 3s < 45 FPS auto-downgrade (Section 7.3)
  const qualityMgr = new QualityManager(
    state.quality,
    stage,
    materials,
    postFX,
    (nextQuality) => {
      state.quality = nextQuality;
      hud.sync();
      hud.showAutoDowngradeToast();
    },
  );

  const animEngine = new AnimationEngine();

  // Hover highlight & click inspection (Section 4.3)
  new InteractionController(
    canvas,
    () => cameraCtrl.activeCamera,
    assembler,
    bus,
    (hoveredId) => {
      state.hoveredPartId = hoveredId;
      renderModeCtrl.setHoveredPart(hoveredId);
    },
  );

  // Dev debug GUI (collapsed by default)
  const gui = new GUI({ title: 'Dev Telemetry' });
  gui.close();
  const guiInfo = {
    drawCalls: 0,
    triangles: 0,
    geometries: 0,
    textures: 0,
    bloom: true,
  };
  gui.add(guiInfo, 'drawCalls').listen().disable();
  gui.add(guiInfo, 'triangles').listen().disable();
  gui.add(guiInfo, 'geometries').listen().disable();
  gui.add(guiInfo, 'textures').listen().disable();
  gui.add(guiInfo, 'bloom').onChange((v: boolean) => {
    postFX.bloomEnabled = v;
  });

  const refFolder = gui.addFolder('Reference Calibration');
  const syncRef = (): void => {
    refOverlay.setState(
      state.referenceOverlay,
      state.referenceOpacity,
      state.cameraPreset,
      state.orthographic,
    );
  };
  refFolder.add(refOverlay.calibration, 'crownY', 40, 120, 1).onChange(syncRef);
  refFolder.add(refOverlay.calibration, 'soleY', 880, 980, 1).onChange(syncRef);
  refFolder.add(refOverlay.calibration, 'offsetXFront', -0.2, 0.2, 0.005).onChange(syncRef);
  refFolder.add(refOverlay.calibration, 'offsetXSide', -0.2, 0.2, 0.005).onChange(syncRef);
  refFolder.add(refOverlay.calibration, 'offsetXBack', -0.2, 0.2, 0.005).onChange(syncRef);
  refFolder.add(refOverlay.calibration, 'offsetY', -0.1, 0.1, 0.005).onChange(syncRef);

  // Wire HUD <-> Engine events
  bus.on('motion:change', (mode) => {
    state.motion = mode;
    animEngine.setMode(mode, rig);
    if (mode === 'Manual') {
      hud.forceRefreshRightDock();
    }
    hud.sync();
  });

  bus.on('playback:toggle', (playing) => {
    state.playing = playing;
    hud.sync();
  });

  bus.on('speed:change', (speed) => {
    state.speed = speed;
    hud.sync();
  });

  bus.on('explode:slider', (s) => {
    state.explodeSlider = s;
    state.explodePreset = 'custom';
    sliderToExplodeTargets(s, state.explodeTarget);
    hud.sync();
  });

  bus.on('explode:preset', (preset) => {
    state.explodePreset = preset;
    const p = EXPLODE_PRESETS[preset];
    state.explodeTarget.t1 = p.t1;
    state.explodeTarget.t2 = p.t2;
    state.explodeTarget.t3 = p.t3;
    state.explodeSlider = Math.max(p.t1, p.t2, p.t3);
    hud.sync();
  });

  bus.on('guidelines:toggle', (visible) => {
    state.showGuideLines = visible;
  });

  bus.on('bloom:toggle', (enabled) => {
    postFX.bloomEnabled = enabled;
    guiInfo.bloom = enabled;
  });

  bus.on('turntable:toggle', (enabled) => {
    state.turntable = enabled;
  });

  bus.on('camera:preset', (preset) => {
    state.cameraPreset = preset;
    if ((preset === '3/4' || preset === 'FREE') && state.orthographic) {
      state.orthographic = false;
      state.referenceOverlay = false;
      cameraCtrl.setOrthographic(false);
    }
    cameraCtrl.setPreset(preset);
    syncRef();
    hud.sync();
  });

  bus.on('camera:ortho', (enabled) => {
    state.orthographic = enabled;
    if (!enabled) {
      state.referenceOverlay = false;
    } else if (state.cameraPreset === '3/4' || state.cameraPreset === 'FREE') {
      state.cameraPreset = 'FRONT';
      cameraCtrl.setPreset('FRONT');
    }
    cameraCtrl.setOrthographic(enabled);
    syncRef();
    hud.sync();
  });

  bus.on('camera:reset', () => {
    state.selectedPartId = null;
    renderModeCtrl.setSelectedPart(null);
    cameraCtrl.clearFocus();
    state.cameraPreset = '3/4';
    state.orthographic = false;
    state.referenceOverlay = false;
    cameraCtrl.setOrthographic(false);
    cameraCtrl.setPreset('3/4');
    syncRef();
    hud.sync();
  });

  bus.on('reference:toggle', (enabled) => {
    state.referenceOverlay = enabled;
    if (enabled) {
      if (state.cameraPreset === '3/4' || state.cameraPreset === 'FREE') {
        state.cameraPreset = 'FRONT';
        cameraCtrl.setPreset('FRONT');
      }
      if (!state.orthographic) {
        state.orthographic = true;
        cameraCtrl.setOrthographic(true);
      }
    }
    syncRef();
    hud.sync();
  });

  bus.on('reference:opacity', (opacity) => {
    state.referenceOpacity = opacity;
    syncRef();
    hud.sync();
  });

  bus.on('renderMode:change', (mode) => {
    state.renderMode = mode;
    renderModeCtrl.setMode(mode);
    hud.sync();
  });

  bus.on('quality:change', (quality) => {
    state.quality = quality;
    qualityMgr.setQuality(quality, window.innerWidth, window.innerHeight, true);
    hud.sync();
  });

  bus.on('part:select', (partId) => {
    state.selectedPartId = partId;
    renderModeCtrl.setSelectedPart(partId);
    if (partId) {
      const part = assembler.getPart(partId);
      if (part) {
        cameraCtrl.focusPart(part);
      }
    } else {
      cameraCtrl.clearFocus();
    }
    hud.sync();
  });

  bus.on('part:hide', (partId) => {
    state.hiddenPartIds.add(partId);
    renderModeCtrl.applyVisibility(state.hiddenPartIds);
    if (state.selectedPartId === partId) {
      state.selectedPartId = null;
      renderModeCtrl.setSelectedPart(null);
      cameraCtrl.clearFocus();
    }
    hud.sync();
  });

  bus.on('part:isolate', (partId) => {
    state.hiddenPartIds.clear();
    for (const part of assembler.partsList) {
      if (part.entry.partId !== partId) {
        state.hiddenPartIds.add(part.entry.partId);
      }
    }
    renderModeCtrl.applyVisibility(state.hiddenPartIds);
    hud.sync();
  });

  bus.on('part:restoreAll', () => {
    state.hiddenPartIds.clear();
    renderModeCtrl.applyVisibility(state.hiddenPartIds);
    hud.sync();
  });

  // Manual joint control (Section 5.2)
  bus.on('manual:angle', ({ jointId, axis, valueDeg }) => {
    const cur = rig.getSemanticAngles(jointId);
    const flex = axis === 'flex' ? valueDeg : cur.flex;
    const abd = axis === 'abd' ? valueDeg : cur.abd;
    const twist = axis === 'twist' ? valueDeg : cur.twist;
    rig.setSemanticAngles(jointId, flex, abd, twist);
    rig.evaluateFollowers();
  });

  bus.on('manual:finger', ({ side, curlDeg }) => {
    rig.setHandCurl(side, curlDeg);
  });

  bus.on('manual:reset', () => {
    rig.resetToBindPose();
    hud.forceRefreshRightDock();
  });

  // P2 High-Resolution 2× PNG Screenshot Export (Section 6.5)
  bus.on('screenshot:export', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const prevPr = renderer.getPixelRatio();
    renderer.setPixelRatio(2.0);
    renderer.setSize(w, h, false);
    stage.resize(w * 2, h * 2);
    postFX.resize(w, h, state.quality);
    postFX.render(cameraCtrl.activeCamera);

    const dataUrl = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `humanoid-robot-${state.motion.toLowerCase()}-${Date.now()}.png`;
    a.click();

    // Restore viewport pixelRatio cleanly
    renderer.setPixelRatio(prevPr);
    renderer.setSize(w, h, false);
    stage.resize(w, h);
    postFX.resize(w, h, state.quality);
  });

  bus.on('hud:toggle', (visible) => {
    state.hudVisible = visible;
    gui.domElement.style.display = visible ? '' : 'none';
    hud.sync();
  });

  // P2 Keyboard Shortcuts: 1-5 Camera, Space Play/Pause, E Cycle Explode, X X-Ray, H HUD
  const explodeCycleOrder: readonly Exclude<ExplodePreset, 'custom'>[] = [
    'assembled',
    'shell_off',
    'full_explode',
  ];
  window.addEventListener('keydown', (e) => {
    const target = e.target as HTMLElement | null;
    if (
      target &&
      (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')
    ) {
      return;
    }

    if (e.key === '1') bus.emit('camera:preset', 'FRONT');
    else if (e.key === '2') bus.emit('camera:preset', 'SIDE');
    else if (e.key === '3') bus.emit('camera:preset', 'BACK');
    else if (e.key === '4') bus.emit('camera:preset', '3/4');
    else if (e.key === '5') bus.emit('camera:preset', 'FREE');
    else if (e.code === 'Space') {
      e.preventDefault();
      bus.emit('playback:toggle', !state.playing);
    } else if (e.key === 'e' || e.key === 'E') {
      const idx =
        state.explodePreset === 'custom'
          ? 0
          : (explodeCycleOrder.indexOf(state.explodePreset) + 1) % explodeCycleOrder.length;
      bus.emit('explode:preset', explodeCycleOrder[idx]!);
    } else if (e.key === 'x' || e.key === 'X') {
      bus.emit('renderMode:change', state.renderMode === 'X-Ray' ? 'PBR' : 'X-Ray');
    } else if (e.key === 'h' || e.key === 'H') {
      bus.emit('hud:toggle', !state.hudVisible);
    }
  });

  // Resize handler
  window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(w, h);
    cameraCtrl.resize(w, h);
    stage.resize(w, h);
    postFX.resize(w, h, state.quality);
  });

  // Expose debug helper
  window.__PROTOTYPE_DEBUG__ = {
    setCamera: (preset, ortho = false) => {
      state.cameraPreset = preset;
      state.orthographic = ortho;
      cameraCtrl.setOrthographic(ortho);
      cameraCtrl.setPreset(preset, true);
      syncRef();
      hud.sync();
    },
    setMotion: (mode, phase) => {
      state.motion = mode;
      animEngine.setMode(mode, rig, true);
      if (phase !== undefined && (mode === 'Walk' || mode === 'Run')) {
        state.playing = false;
        animEngine.evaluateGaitAtPhase(mode, phase, rig);
      }
      hud.sync();
    },
    setExplode: (s, snap = true) => {
      if (typeof s === 'number') {
        state.explodeSlider = s;
        state.explodePreset = 'custom';
        sliderToExplodeTargets(s, state.explodeTarget);
      } else {
        state.explodePreset = s;
        const p = EXPLODE_PRESETS[s];
        state.explodeTarget.t1 = p.t1;
        state.explodeTarget.t2 = p.t2;
        state.explodeTarget.t3 = p.t3;
        state.explodeSlider = Math.max(p.t1, p.t2, p.t3);
      }
      if (snap) {
        explodeCtrl.update(
          0,
          state.explodeTarget,
          state.explodeCurrent,
          rig,
          state.showGuideLines,
          true,
        );
      }
      hud.sync();
    },
    setRenderMode: (mode) => {
      state.renderMode = mode;
      renderModeCtrl.setMode(mode);
      hud.sync();
    },
    setQuality: (quality) => {
      bus.emit('quality:change', quality);
    },
    setReferenceOverlay: (enabled, opacity) => {
      state.referenceOverlay = enabled;
      if (opacity !== undefined) state.referenceOpacity = opacity;
      syncRef();
      hud.sync();
    },
    selectPart: (partId) => {
      bus.emit('part:select', partId);
    },
    setBloom: (enabled) => {
      postFX.bloomEnabled = enabled;
      guiInfo.bloom = enabled;
    },
    setHudVisible: (visible) => {
      bus.emit('hud:toggle', visible);
    },
    setGlbOverrides: async (enabled) => {
      registry.useGlbOverrides = enabled;
      const allIds = assembler.partsList.map((p) => p.entry.partId);
      if (enabled) {
        const loaded = await registry.loadGlbOverrides(allIds);
        if (loaded.length > 0) {
          assembler.applyPartOverrides(registry, materials, loaded);
          renderModeCtrl.refreshAllMaterials();
        }
        return loaded;
      }
      assembler.applyPartOverrides(registry, materials, allIds);
      renderModeCtrl.refreshAllMaterials();
      return [];
    },
    getDrawCalls: () => postFX.lastMainDrawCalls,
    getMemory: () => ({
      geometries: renderer.info.memory.geometries,
      textures: renderer.info.memory.textures,
    }),
  };

  // Main render loop (zero heap allocations inside RAF)
  let lastTime = performance.now();
  let fpsFrames = 0;
  let fpsAccum = 0;
  let turntableAngle = 0;

  function animate(now: number): void {
    requestAnimationFrame(animate);

    const dt = Math.min(0.05, Math.max(0.001, (now - lastTime) / 1000));
    lastTime = now;

    if (state.turntable) {
      turntableAngle = (turntableAngle + dt * 0.45) % (Math.PI * 2);
      rig.setTurntableYaw(turntableAngle);
    } else if (turntableAngle !== 0) {
      turntableAngle = 0;
      rig.setTurntableYaw(0);
    }

    // 1. Advance explode springs & apply root floatY BEFORE IK so targets stay locked to floated root
    explodeCtrl.stepSpringAndFloat(dt, state.explodeTarget, state.explodeCurrent, rig);

    // 2. Animation -> limits -> IK -> followers
    animEngine.update(dt, state.speed, state.playing, rig);

    // 3. Apply explode container offsets & update 3D assembly guide lines
    explodeCtrl.applyOffsetsAndGuideLines(state.explodeCurrent, rig, state.showGuideLines);

    // 4. Sync LED emissive intensity
    materials.grades.E.emissiveIntensity = animEngine.coreEmissiveIntensity;

    // 5. Stage & Camera
    stage.update(dt, animEngine.treadmillSpeed);
    cameraCtrl.update(dt);

    // 6. Selective Bloom + Final ACES render
    postFX.render(cameraCtrl.activeCamera);

    // Telemetry & Auto-Downgrade sampling every ~0.4s
    fpsFrames += 1;
    fpsAccum += dt;
    if (fpsAccum >= 0.4) {
      state.fps = fpsFrames / fpsAccum;
      state.drawCalls = postFX.lastMainDrawCalls;
      hud.updateTelemetry(state.fps);
      qualityMgr.sampleFps(state.fps, fpsAccum, window.innerWidth, window.innerHeight);

      guiInfo.drawCalls = state.drawCalls;
      guiInfo.triangles = postFX.lastMainTriangles;
      guiInfo.geometries = renderer.info.memory.geometries;
      guiInfo.textures = renderer.info.memory.textures;
      fpsFrames = 0;
      fpsAccum = 0;
    }
  }

  requestAnimationFrame(animate);
}

bootstrap();
