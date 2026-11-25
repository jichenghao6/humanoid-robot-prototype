import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import type { QualityLevel } from '../core/AppState.ts';
import { BLOOM_LAYER } from '../robot/RobotAssembler.ts';

const BLOOM_MIX_VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const BLOOM_MIX_FRAGMENT = /* glsl */ `
  uniform sampler2D baseTexture;
  uniform sampler2D bloomTexture;
  uniform float bloomMultiplier;
  varying vec2 vUv;
  void main() {
    vec4 baseColor = texture2D(baseTexture, vUv);
    vec4 bloomColor = texture2D(bloomTexture, vUv);
    gl_FragColor = vec4(baseColor.rgb + bloomColor.rgb * bloomMultiplier, baseColor.a);
  }
`;

export class SelectiveBloomPipeline {
  public bloomEnabled = true;
  /** True draw calls & triangles of the main scene RenderPass (captured before post passes reset renderer.info) */
  public lastMainDrawCalls = 0;
  public lastMainTriangles = 0;

  private readonly bloomLayer = new THREE.Layers();
  private readonly savedMaterials = new Map<THREE.Object3D, THREE.Material | THREE.Material[]>();
  private readonly hiddenTransparentInBloom: THREE.Object3D[] = [];
  private readonly bloomRenderTarget: THREE.WebGLRenderTarget;
  private finalRenderTarget: THREE.WebGLRenderTarget;
  private readonly bloomComposer: EffectComposer;
  private readonly finalComposer: EffectComposer;
  private readonly bloomRenderPass: RenderPass;
  private readonly finalRenderPass: RenderPass;
  private readonly bloomPass: UnrealBloomPass;
  private readonly mixPass: ShaderPass;
  private readonly outputPass: OutputPass;

  private readonly darkenVisitor: (obj: THREE.Object3D) => void;
  private readonly restoreVisitor: (obj: THREE.Object3D) => void;
  private readonly hiddenDuringBloom: THREE.Object3D[] = [];
  private readonly hiddenSavedState: boolean[] = [];

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    private readonly scene: THREE.Scene,
    initialCamera: THREE.Camera,
    private readonly darkMask: THREE.MeshBasicMaterial,
    width: number,
    height: number,
  ) {
    this.bloomLayer.set(BLOOM_LAYER);

    const pixelRatio = renderer.getPixelRatio();
    const rtW = Math.max(2, Math.floor(width * pixelRatio));
    const rtH = Math.max(2, Math.floor(height * pixelRatio));

    this.bloomRenderTarget = new THREE.WebGLRenderTarget(rtW, rtH, {
      type: THREE.HalfFloatType,
      samples: 0,
    });
    this.finalRenderTarget = new THREE.WebGLRenderTarget(rtW, rtH, {
      type: THREE.HalfFloatType,
      samples: 4,
    });

    // Bloom-only composer (threshold 0, strength 0.9, radius 0.35 — Section 6.3)
    this.bloomComposer = new EffectComposer(renderer, this.bloomRenderTarget);
    this.bloomComposer.renderToScreen = false;
    this.bloomRenderPass = new RenderPass(scene, initialCamera);
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(width, height), 0.9, 0.35, 0.0);
    this.bloomComposer.addPass(this.bloomRenderPass);
    this.bloomComposer.addPass(this.bloomPass);

    // Final composer: base render + additive selective bloom + ACES OutputPass
    this.finalComposer = new EffectComposer(renderer, this.finalRenderTarget);
    this.finalRenderPass = new RenderPass(scene, initialCamera);
    const origRender = this.finalRenderPass.render.bind(this.finalRenderPass);
    this.finalRenderPass.render = (r, writeBuf, readBuf, dt, maskActive): void => {
      origRender(r, writeBuf, readBuf, dt, maskActive);
      this.lastMainDrawCalls = r.info.render.calls;
      this.lastMainTriangles = r.info.render.triangles;
    };
    this.mixPass = new ShaderPass(
      new THREE.ShaderMaterial({
        uniforms: {
          baseTexture: { value: null },
          bloomTexture: { value: this.bloomComposer.renderTarget2.texture },
          bloomMultiplier: { value: 1.0 },
        },
        vertexShader: BLOOM_MIX_VERTEX,
        fragmentShader: BLOOM_MIX_FRAGMENT,
        defines: {},
      }),
      'baseTexture',
    );
    this.mixPass.needsSwap = true;
    this.outputPass = new OutputPass();

    this.finalComposer.addPass(this.finalRenderPass);
    this.finalComposer.addPass(this.mixPass);
    this.finalComposer.addPass(this.outputPass);

    // Pre-bound zero-allocation traverse visitors
    this.darkenVisitor = (obj: THREE.Object3D): void => {
      if (obj instanceof THREE.Mesh && !this.bloomLayer.test(obj.layers)) {
        const mat = obj.material;
        const single = (Array.isArray(mat) ? mat[0] : mat) as
          | (THREE.Material & { wireframe?: boolean })
          | undefined;
        if (single && (single.wireframe || (single.transparent && !single.depthWrite))) {
          if (obj.visible) {
            this.hiddenTransparentInBloom.push(obj);
            obj.visible = false;
          }
        } else {
          this.savedMaterials.set(obj, obj.material);
          obj.material = this.darkMask;
        }
      }
    };

    this.restoreVisitor = (obj: THREE.Object3D): void => {
      const orig = this.savedMaterials.get(obj);
      if (orig !== undefined && obj instanceof THREE.Mesh) {
        obj.material = orig;
      }
    };
  }

  /**
   * Registers scene objects (e.g., Reflector, assembly guide lines) to hide during the bloom pass.
   */
  public excludeFromBloomPass(obj: THREE.Object3D): void {
    this.hiddenDuringBloom.push(obj);
    this.hiddenSavedState.push(true);
  }

  public setCamera(camera: THREE.Camera): void {
    this.bloomRenderPass.camera = camera;
    this.finalRenderPass.camera = camera;
  }

  public setQuality(quality: QualityLevel, width: number, height: number): void {
    const scale = quality === 'High' ? 1.0 : 0.5;
    const pr = this.renderer.getPixelRatio();
    const desiredSamples = quality === 'High' ? 4 : 0;
    if (this.finalRenderTarget.samples !== desiredSamples) {
      const nextRt = new THREE.WebGLRenderTarget(
        Math.max(2, Math.floor(width * pr)),
        Math.max(2, Math.floor(height * pr)),
        {
          type: THREE.HalfFloatType,
          samples: desiredSamples,
        },
      );
      this.finalComposer.reset(nextRt);
      this.finalRenderTarget = nextRt;
    }
    this.bloomComposer.setPixelRatio(pr * scale);
    this.bloomComposer.setSize(width, height);
    this.finalComposer.setPixelRatio(pr);
    this.finalComposer.setSize(width, height);
  }

  public resize(width: number, height: number, quality: QualityLevel): void {
    this.setQuality(quality, width, height);
  }

  public render(camera: THREE.Camera): void {
    this.setCamera(camera);

    const mixMat = this.mixPass.material as THREE.ShaderMaterial;
    const multUniform = mixMat.uniforms.bloomMultiplier;

    if (!this.bloomEnabled) {
      if (multUniform) multUniform.value = 0.0;
      this.finalComposer.render();
      return;
    }

    if (multUniform) multUniform.value = 1.0;

    // 1. Darken non-bloom meshes, hide excluded/transparent overlays, clear background & fog for pure bloom pass
    const savedBg = this.scene.background;
    const savedFog = this.scene.fog;
    this.scene.background = null;
    this.scene.fog = null;

    for (let i = 0; i < this.hiddenDuringBloom.length; i++) {
      const obj = this.hiddenDuringBloom[i]!;
      this.hiddenSavedState[i] = obj.visible;
      obj.visible = false;
    }

    this.scene.traverse(this.darkenVisitor);

    // 2. Render selective bloom pass
    this.bloomComposer.render();

    // 3. Restore original materials, excluded objects, background & fog
    this.scene.traverse(this.restoreVisitor);
    this.savedMaterials.clear();
    for (let i = 0; i < this.hiddenTransparentInBloom.length; i++) {
      this.hiddenTransparentInBloom[i]!.visible = true;
    }
    this.hiddenTransparentInBloom.length = 0;

    for (let i = 0; i < this.hiddenDuringBloom.length; i++) {
      this.hiddenDuringBloom[i]!.visible = this.hiddenSavedState[i]!;
    }
    this.scene.background = savedBg;
    this.scene.fog = savedFog;

    // 4. Render final composited frame (base + selective bloom + ACES OutputPass)
    this.finalComposer.render();
  }

  public dispose(): void {
    this.bloomRenderTarget.dispose();
    this.finalRenderTarget.dispose();
    this.bloomPass.dispose();
    (this.mixPass.material as THREE.ShaderMaterial).dispose();
    this.outputPass.dispose();
    this.bloomComposer.dispose();
    this.finalComposer.dispose();
  }
}
