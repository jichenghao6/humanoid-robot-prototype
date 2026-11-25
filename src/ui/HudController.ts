import { PARTS_CATALOG_MAP, computeCatalogStats } from '../config/partsCatalog.ts';
import { ROBOT_TOTAL_HEIGHT } from '../config/proportions.ts';
import {
  RIG_JOINT_MAP,
  computeRigDofStats,
  type SemanticAxis,
} from '../config/rig.ts';
import type {
  AppStateData,
  CameraPreset,
  ExplodePreset,
  MotionMode,
  RenderMode,
} from '../core/AppState.ts';
import type { EventBus } from '../core/EventBus.ts';
import type { RobotRig } from '../robot/RobotRig.ts';

interface ManualGroupSpec {
  title: string;
  open?: boolean;
  joints: ReadonlyArray<{ id: string; label: string }>;
}

const MANUAL_GROUPS: readonly ManualGroupSpec[] = [
  {
    title: '头部与颈部 / Head & Neck',
    open: true,
    joints: [
      { id: 'neck', label: '颈部 Neck' },
      { id: 'head', label: '头部 Head' },
    ],
  },
  {
    title: '腰腹与骨盆 / Spine & Pelvis',
    joints: [
      { id: 'pelvis', label: '骨盆 Pelvis' },
      { id: 'spine_lower', label: '下腰 SpineLower' },
      { id: 'spine_upper', label: '上背 SpineUpper' },
    ],
  },
  {
    title: '右臂 / Right Arm',
    open: true,
    joints: [
      { id: 'shoulder_R', label: '右肩 Shoulder_R' },
      { id: 'elbow_R', label: '右肘 Elbow_R' },
      { id: 'wrist_R', label: '右腕 Wrist_R' },
    ],
  },
  {
    title: '左臂 / Left Arm',
    joints: [
      { id: 'shoulder_L', label: '左肩 Shoulder_L' },
      { id: 'elbow_L', label: '左肘 Elbow_L' },
      { id: 'wrist_L', label: '左腕 Wrist_L' },
    ],
  },
  {
    title: '右腿 / Right Leg',
    joints: [
      { id: 'hip_R', label: '右髋 Hip_R' },
      { id: 'knee_R', label: '右膝 Knee_R' },
      { id: 'ankle_R', label: '右踝 Ankle_R' },
      { id: 'toe_R', label: '右趾 Toe_R' },
    ],
  },
  {
    title: '左腿 / Left Leg',
    joints: [
      { id: 'hip_L', label: '左髋 Hip_L' },
      { id: 'knee_L', label: '左膝 Knee_L' },
      { id: 'ankle_L', label: '左踝 Ankle_L' },
      { id: 'toe_L', label: '左趾 Toe_L' },
    ],
  },
];

const AXIS_LABELS: Record<SemanticAxis, string> = {
  flex: '屈伸 Flex',
  abd: '外展/侧倾 Abd',
  twist: '扭转 Twist',
};

export class HudController {
  private readonly root: HTMLElement;
  private fpsEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private qualityBtn!: HTMLButtonElement;
  private explodeValEl!: HTMLElement;
  private explodeInput!: HTMLInputElement;
  private speedValEl!: HTMLElement;
  private speedInput!: HTMLInputElement;
  private playBtn!: HTMLButtonElement;
  private orthoBtn!: HTMLButtonElement;
  private leftDockEl!: HTMLElement;
  private rightDockEl!: HTMLElement;
  private refWrapEl!: HTMLElement;
  private refBtn!: HTMLButtonElement;
  private refOpacityInput!: HTMLInputElement;
  private refOpacityValEl!: HTMLElement;
  private toastEl!: HTMLElement;

  private motionButtons = new Map<MotionMode, HTMLButtonElement>();
  private cameraButtons = new Map<CameraPreset, HTMLButtonElement>();
  private presetButtons = new Map<Exclude<ExplodePreset, 'custom'>, HTMLButtonElement>();
  private renderModeButtons = new Map<RenderMode, HTMLButtonElement>();

  private referenceAvailable = false;
  private lastRightDockKey = '';

  constructor(
    container: HTMLElement,
    private readonly state: AppStateData,
    private readonly bus: EventBus,
    private readonly rig: RobotRig,
  ) {
    this.root = container;
    this.renderShell();
    this.bindEvents();
    this.sync();
  }

  public setReferenceAvailable(available: boolean): void {
    this.referenceAvailable = available;
    this.refWrapEl.style.display = available ? 'flex' : 'none';
  }

  public showAutoDowngradeToast(): void {
    this.toastEl.innerHTML = `
      <span>检测到帧率持续低于 45 FPS，已自动切换至 Low 档</span>
      <button class="hud-btn active" id="hud-toast-restore-high">切回 High 档</button>
      <button class="hud-btn" id="hud-toast-close">关闭</button>
    `;
    this.toastEl.classList.add('visible');
    this.toastEl.querySelector('#hud-toast-restore-high')!.addEventListener('click', () => {
      this.toastEl.classList.remove('visible');
      this.bus.emit('quality:change', 'High');
    });
    this.toastEl.querySelector('#hud-toast-close')!.addEventListener('click', () => {
      this.toastEl.classList.remove('visible');
    });
  }

  public forceRefreshRightDock(): void {
    this.lastRightDockKey = '';
    this.renderRightDock();
  }

  private renderShell(): void {
    this.root.innerHTML = `
      <header class="hud-panel hud-topbar">
        <div class="hud-brand">
          <span class="hud-slashes">///</span>
          <div>
            <div class="hud-title">HUMANOID ROBOT // DIGITAL PROTOTYPE</div>
            <div class="hud-subtitle">INTELLIGENT · ADAPTIVE · HUMAN-CENTRIC · FUTURE-READY</div>
          </div>
        </div>
        <div class="hud-telemetry">
          <span class="hud-badge" id="hud-status">IDLE · ASSEMBLED</span>
          <span id="hud-fps">60 FPS</span>
          <button class="hud-btn" id="hud-quality-btn" title="切换 High / Low 画质档位">HIGH</button>
          <button class="hud-btn" id="hud-shot-btn" title="导出 2× 高清 PNG 截图">2× 截图</button>
          <button class="hud-btn" id="hud-hide-btn" title="快捷键 H">隐藏 HUD</button>
        </div>
      </header>

      <div class="hud-panel hud-toast" id="hud-toast"></div>

      <button class="hud-panel hud-btn hud-drawer-toggle hud-drawer-toggle-left" id="hud-drawer-left-btn">控制面板</button>
      <button class="hud-panel hud-btn hud-drawer-toggle hud-drawer-toggle-right" id="hud-drawer-right-btn">检视面板</button>

      <aside class="hud-panel hud-left" id="hud-left-dock">
        <div>
          <div class="hud-section-title">运动模式 / Motion</div>
          <div class="hud-btn-grid">
            <button class="hud-btn" data-motion="Idle">Idle</button>
            <button class="hud-btn" data-motion="Walk">Walk</button>
            <button class="hud-btn" data-motion="Run">Run</button>
            <button class="hud-btn" data-motion="Jump">Jump</button>
            <button class="hud-btn" data-motion="Wave">Wave</button>
            <button class="hud-btn" data-motion="Manual">Manual</button>
          </div>
        </div>

        <div class="hud-slider-row">
          <div class="hud-slider-header">
            <span>播放速度</span>
            <span id="hud-speed-val">1.0x</span>
          </div>
          <div class="hud-btn-row">
            <button class="hud-btn" id="hud-play-btn" style="flex:0 0 64px">暂停</button>
            <input class="hud-slider" id="hud-speed-slider" type="range" min="0.2" max="2.0" step="0.1" value="1.0" style="flex:1" />
          </div>
        </div>

        <div class="hud-slider-row">
          <div class="hud-slider-header">
            <span>三层爆炸拆解</span>
            <span id="hud-explode-val">0%</span>
          </div>
          <input class="hud-slider" id="hud-explode-slider" type="range" min="0" max="100" step="1" value="0" />
          <div class="hud-btn-grid" style="margin-top:4px">
            <button class="hud-btn" data-preset="assembled">完整组装</button>
            <button class="hud-btn" data-preset="shell_off">剥离外甲</button>
            <button class="hud-btn" data-preset="full_explode">全爆炸</button>
          </div>
        </div>

        <div style="display:flex;flex-direction:column;gap:7px">
          <label class="hud-checkbox">
            <input type="checkbox" id="hud-guidelines-chk" checked />
            <span>3D 装配引导线</span>
          </label>
          <label class="hud-checkbox">
            <input type="checkbox" id="hud-bloom-chk" checked />
            <span>选择性 Bloom（仅 LED）</span>
          </label>
          <label class="hud-checkbox">
            <input type="checkbox" id="hud-turntable-chk" />
            <span>旋转展示台</span>
          </label>
        </div>

        <div class="hud-shortcuts">
          快捷键：1-5 机位 · Space 播放 · E 拆解 · X 透视 · H 隐藏
        </div>
      </aside>

      <aside class="hud-panel hud-right" id="hud-right-dock"></aside>

      <nav class="hud-panel hud-bottom">
        <button class="hud-btn" data-cam="FRONT">FRONT</button>
        <button class="hud-btn" data-cam="SIDE">SIDE</button>
        <button class="hud-btn" data-cam="BACK">BACK</button>
        <button class="hud-btn" data-cam="3/4">3/4</button>
        <button class="hud-btn" data-cam="FREE">FREE</button>
        <button class="hud-btn" id="hud-ortho-btn">正交</button>
        <div class="hud-divider"></div>
        <div class="hud-ref-wrap" id="hud-ref-wrap" style="display:none">
          <button class="hud-btn" id="hud-ref-btn">参考图叠加</button>
          <input class="hud-slider" id="hud-ref-opacity" type="range" min="0" max="100" step="5" value="45" style="width:66px" />
          <span id="hud-ref-opacity-val" style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9cc4ec">45%</span>
          <div class="hud-divider"></div>
        </div>
        <button class="hud-btn" data-render="PBR">PBR</button>
        <button class="hud-btn" data-render="X-Ray">X-Ray</button>
        <button class="hud-btn" data-render="Wireframe">Wireframe</button>
      </nav>

      <div class="hud-ruler"></div>
      <button class="hud-panel hud-btn hud-restore-btn" id="hud-restore-btn">显示 HUD (H)</button>
    `;

    this.fpsEl = this.root.querySelector('#hud-fps')!;
    this.statusEl = this.root.querySelector('#hud-status')!;
    this.qualityBtn = this.root.querySelector('#hud-quality-btn')!;
    this.explodeValEl = this.root.querySelector('#hud-explode-val')!;
    this.explodeInput = this.root.querySelector('#hud-explode-slider')!;
    this.speedValEl = this.root.querySelector('#hud-speed-val')!;
    this.speedInput = this.root.querySelector('#hud-speed-slider')!;
    this.playBtn = this.root.querySelector('#hud-play-btn')!;
    this.orthoBtn = this.root.querySelector('#hud-ortho-btn')!;
    this.leftDockEl = this.root.querySelector('#hud-left-dock')!;
    this.rightDockEl = this.root.querySelector('#hud-right-dock')!;
    this.refWrapEl = this.root.querySelector('#hud-ref-wrap')!;
    this.refBtn = this.root.querySelector('#hud-ref-btn')!;
    this.refOpacityInput = this.root.querySelector('#hud-ref-opacity')!;
    this.refOpacityValEl = this.root.querySelector('#hud-ref-opacity-val')!;
    this.toastEl = this.root.querySelector('#hud-toast')!;

    this.root.querySelectorAll<HTMLButtonElement>('[data-motion]').forEach((btn) => {
      this.motionButtons.set(btn.dataset.motion as MotionMode, btn);
    });
    this.root.querySelectorAll<HTMLButtonElement>('[data-cam]').forEach((btn) => {
      this.cameraButtons.set(btn.dataset.cam as CameraPreset, btn);
    });
    this.root.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach((btn) => {
      this.presetButtons.set(btn.dataset.preset as Exclude<ExplodePreset, 'custom'>, btn);
    });
    this.root.querySelectorAll<HTMLButtonElement>('[data-render]').forEach((btn) => {
      this.renderModeButtons.set(btn.dataset.render as RenderMode, btn);
    });
  }

  private renderRightDock(): void {
    const key =
      this.state.motion === 'Manual'
        ? `MANUAL:${this.state.selectedPartId ?? ''}`
        : this.state.selectedPartId
          ? `PART:${this.state.selectedPartId}:${this.state.hiddenPartIds.size}`
          : `SUMMARY:${this.state.hiddenPartIds.size}`;
    if (key === this.lastRightDockKey) return;
    this.lastRightDockKey = key;

    // 1. Manual Joint Control Mode (Section 5.2 & 6.5)
    if (this.state.motion === 'Manual') {
      this.renderManualControls();
      return;
    }

    // 2. Clicked Part Engineering Inspector Card (Section 4.3.3)
    const selectedId = this.state.selectedPartId;
    const selectedEntry = selectedId ? PARTS_CATALOG_MAP.get(selectedId) : undefined;
    const hiddenCount = this.state.hiddenPartIds.size;

    if (selectedEntry) {
      const tierLabel =
        selectedEntry.tier === 1
          ? 'Tier 1 外装甲层'
          : selectedEntry.tier === 2
            ? 'Tier 2 关节致动器'
            : 'Tier 3 内部骨架/核心';
      const isHidden = this.state.hiddenPartIds.has(selectedEntry.partId);

      this.rightDockEl.innerHTML = `
        <div class="hud-inspector-card">
          <div class="hud-section-title">工程检视卡 / Inspector</div>
          <div class="hud-inspector-header">
            <div>
              <div class="hud-inspector-name">${selectedEntry.name}</div>
              <div class="hud-inspector-id">${selectedEntry.partId}</div>
            </div>
            <span class="hud-badge">T${selectedEntry.tier}</span>
          </div>
          <div class="hud-stat-grid">
            <div class="hud-stat">
              <div class="hud-stat-label">层级</div>
              <div class="hud-stat-val" style="font-size:11px">${tierLabel}</div>
            </div>
            <div class="hud-stat">
              <div class="hud-stat-label">父关节</div>
              <div class="hud-stat-val" style="font-size:11px">${selectedEntry.parentJoint}</div>
            </div>
            <div class="hud-stat">
              <div class="hud-stat-label">材质等级</div>
              <div class="hud-stat-val">${selectedEntry.materials.join(' / ')}</div>
            </div>
            <div class="hud-stat">
              <div class="hud-stat-label">单件质量</div>
              <div class="hud-stat-val">${selectedEntry.massKg.toFixed(2)} kg</div>
            </div>
          </div>
          <div class="hud-inspector-desc">${selectedEntry.description}</div>
          <div class="hud-inspector-actions">
            <button class="hud-btn" id="hud-act-hide">${isHidden ? '已隐藏' : '隐藏此零件'}</button>
            <button class="hud-btn" id="hud-act-isolate">仅显示此零件</button>
            <button class="hud-btn" id="hud-act-restore">恢复全部${hiddenCount > 0 ? ` (${hiddenCount})` : ''}</button>
            <button class="hud-btn" id="hud-act-reset">重置视图</button>
          </div>
        </div>
      `;

      this.rightDockEl.querySelector('#hud-act-hide')!.addEventListener('click', () => {
        this.bus.emit('part:hide', selectedEntry.partId);
      });
      this.rightDockEl.querySelector('#hud-act-isolate')!.addEventListener('click', () => {
        this.bus.emit('part:isolate', selectedEntry.partId);
      });
      this.rightDockEl.querySelector('#hud-act-restore')!.addEventListener('click', () => {
        this.bus.emit('part:restoreAll', undefined);
      });
      this.rightDockEl.querySelector('#hud-act-reset')!.addEventListener('click', () => {
        this.bus.emit('camera:reset', undefined);
      });
      return;
    }

    // 3. Default Whole-Robot Engineering Summary (dynamically computed from catalog & rig)
    const catalogStats = computeCatalogStats();
    const dofStats = computeRigDofStats();

    this.rightDockEl.innerHTML = `
      <div class="hud-section-title">整机工程摘要 / Spec (Catalog)</div>
      <div class="hud-stat-grid">
        <div class="hud-stat">
          <div class="hud-stat-label">身高</div>
          <div class="hud-stat-val">${ROBOT_TOTAL_HEIGHT.toFixed(3)} m</div>
        </div>
        <div class="hud-stat">
          <div class="hud-stat-label">独立部件</div>
          <div class="hud-stat-val">${catalogStats.totalParts} pcs</div>
        </div>
        <div class="hud-stat">
          <div class="hud-stat-label">主动 DOF</div>
          <div class="hud-stat-val">${dofStats.totalActiveDof} (+${dofStats.fingerDof}指)</div>
        </div>
        <div class="hud-stat">
          <div class="hud-stat-label">整机质量</div>
          <div class="hud-stat-val">${catalogStats.totalMassKg.toFixed(1)} kg</div>
        </div>
      </div>

      <div>
        <div class="hud-section-title">拆解层级统计 / Tier</div>
        <div class="hud-tier-legend">
          <div class="hud-tier-item">
            <span><i class="hud-tier-swatch" style="background:#e6ecf5"></i>Tier 1 外装甲层</span>
            <span>${catalogStats.tierCounts[1]}</span>
          </div>
          <div class="hud-tier-item">
            <span><i class="hud-tier-swatch" style="background:#3b495e"></i>Tier 2 关节致动器</span>
            <span>${catalogStats.tierCounts[2]}</span>
          </div>
          <div class="hud-tier-item">
            <span><i class="hud-tier-swatch" style="background:#d48a4a"></i>Tier 3 内部骨架/核心</span>
            <span>${catalogStats.tierCounts[3]}</span>
          </div>
        </div>
      </div>

      <div style="font-size:10px;color:rgba(156,192,228,0.72);line-height:1.45">
        提示：点击任意 3D 零件可聚焦检视；切换 Manual 可手动微调各关节角度。
      </div>
      ${
        hiddenCount > 0
          ? `<button class="hud-btn active" id="hud-quick-restore">恢复已隐藏零件 (${hiddenCount})</button>`
          : ''
      }
    `;

    this.rightDockEl.querySelector('#hud-quick-restore')?.addEventListener('click', () => {
      this.bus.emit('part:restoreAll', undefined);
    });
  }

  private renderManualControls(): void {
    const groupsHtml: string[] = [];

    for (const group of MANUAL_GROUPS) {
      const sliders: string[] = [];
      for (const j of group.joints) {
        const def = RIG_JOINT_MAP.get(j.id);
        if (!def) continue;
        const cur = this.rig.getSemanticAngles(j.id);
        const axes: SemanticAxis[] = ['flex', 'abd', 'twist'];
        for (const axis of axes) {
          const range = def.limits[axis];
          if (!range) continue;
          const val = Math.round(cur[axis]);
          sliders.push(`
            <div class="hud-slider-row">
              <div class="hud-slider-header">
                <span>${j.label} · ${AXIS_LABELS[axis]}</span>
                <span data-val-for="${j.id}:${axis}">${val}°</span>
              </div>
              <input
                class="hud-slider"
                type="range"
                min="${range.min}"
                max="${range.max}"
                step="1"
                value="${val}"
                data-joint="${j.id}"
                data-axis="${axis}"
              />
            </div>
          `);
        }
      }
      groupsHtml.push(`
        <details class="hud-manual-group" ${group.open ? 'open' : ''}>
          <summary>${group.title}</summary>
          <div class="hud-manual-group-body">${sliders.join('')}</div>
        </details>
      `);
    }

    // P2 Finger grip sliders (L & R)
    const curFingerL = Math.round(this.rig.getSemanticAngles('finger_index_L').flex);
    const curFingerR = Math.round(this.rig.getSemanticAngles('finger_index_R').flex);
    groupsHtml.push(`
      <details class="hud-manual-group">
        <summary>五指握持 / Fingers (P2)</summary>
        <div class="hud-manual-group-body">
          <div class="hud-slider-row">
            <div class="hud-slider-header">
              <span>右手握拳 Curl_R</span>
              <span data-finger-val="R">${curFingerR}°</span>
            </div>
            <input class="hud-slider" type="range" min="0" max="90" step="1" value="${curFingerR}" data-finger-side="R" />
          </div>
          <div class="hud-slider-row">
            <div class="hud-slider-header">
              <span>左手握拳 Curl_L</span>
              <span data-finger-val="L">${curFingerL}°</span>
            </div>
            <input class="hud-slider" type="range" min="0" max="90" step="1" value="${curFingerL}" data-finger-side="L" />
          </div>
        </div>
      </details>
    `);

    const selectedEntry = this.state.selectedPartId
      ? PARTS_CATALOG_MAP.get(this.state.selectedPartId)
      : undefined;
    const focusBannerHtml = selectedEntry
      ? `
        <div class="hud-manual-focus-banner">
          <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">聚焦: ${selectedEntry.name}</span>
          <button class="hud-btn" id="hud-manual-unfocus" style="padding:2px 6px;font-size:10px">取消</button>
        </div>
      `
      : '';

    this.rightDockEl.innerHTML = `
      <div class="hud-manual-header">
        <div class="hud-section-title" style="margin:0">手动关节 / Manual DOF</div>
        <button class="hud-btn active" id="hud-manual-reset">一键归零</button>
      </div>
      ${focusBannerHtml}
      ${groupsHtml.join('')}
    `;

    this.rightDockEl.querySelector('#hud-manual-unfocus')?.addEventListener('click', () => {
      this.bus.emit('part:select', null);
    });

    this.rightDockEl.querySelector('#hud-manual-reset')!.addEventListener('click', () => {
      this.bus.emit('manual:reset', undefined);
    });

    this.rightDockEl.querySelectorAll<HTMLInputElement>('[data-joint]').forEach((input) => {
      input.addEventListener('input', () => {
        const jointId = input.dataset.joint!;
        const axis = input.dataset.axis as SemanticAxis;
        const valueDeg = Number(input.value);
        const label = this.rightDockEl.querySelector(`[data-val-for="${jointId}:${axis}"]`);
        if (label) label.textContent = `${Math.round(valueDeg)}°`;
        this.bus.emit('manual:angle', { jointId, axis, valueDeg });
      });
    });

    this.rightDockEl.querySelectorAll<HTMLInputElement>('[data-finger-side]').forEach((input) => {
      input.addEventListener('input', () => {
        const side = input.dataset.fingerSide as 'L' | 'R';
        const curlDeg = Number(input.value);
        const label = this.rightDockEl.querySelector(`[data-finger-val="${side}"]`);
        if (label) label.textContent = `${Math.round(curlDeg)}°`;
        this.bus.emit('manual:finger', { side, curlDeg });
      });
    });
  }

  private bindEvents(): void {
    for (const [mode, btn] of this.motionButtons) {
      btn.addEventListener('click', () => this.bus.emit('motion:change', mode));
    }
    for (const [cam, btn] of this.cameraButtons) {
      btn.addEventListener('click', () => this.bus.emit('camera:preset', cam));
    }
    for (const [preset, btn] of this.presetButtons) {
      btn.addEventListener('click', () => this.bus.emit('explode:preset', preset));
    }
    for (const [rm, btn] of this.renderModeButtons) {
      btn.addEventListener('click', () => this.bus.emit('renderMode:change', rm));
    }

    this.playBtn.addEventListener('click', () => {
      this.bus.emit('playback:toggle', !this.state.playing);
    });

    this.speedInput.addEventListener('input', () => {
      this.bus.emit('speed:change', Number(this.speedInput.value));
    });

    this.explodeInput.addEventListener('input', () => {
      this.bus.emit('explode:slider', Number(this.explodeInput.value) / 100);
    });

    this.root
      .querySelector<HTMLInputElement>('#hud-guidelines-chk')!
      .addEventListener('change', (e) => {
        this.bus.emit('guidelines:toggle', (e.target as HTMLInputElement).checked);
      });

    this.root
      .querySelector<HTMLInputElement>('#hud-bloom-chk')!
      .addEventListener('change', (e) => {
        this.bus.emit('bloom:toggle', (e.target as HTMLInputElement).checked);
      });

    this.root
      .querySelector<HTMLInputElement>('#hud-turntable-chk')!
      .addEventListener('change', (e) => {
        this.bus.emit('turntable:toggle', (e.target as HTMLInputElement).checked);
      });

    this.orthoBtn.addEventListener('click', () => {
      this.bus.emit('camera:ortho', !this.state.orthographic);
    });

    this.refBtn.addEventListener('click', () => {
      this.bus.emit('reference:toggle', !this.state.referenceOverlay);
    });

    this.refOpacityInput.addEventListener('input', () => {
      this.bus.emit('reference:opacity', Number(this.refOpacityInput.value) / 100);
    });

    this.qualityBtn.addEventListener('click', () => {
      this.bus.emit('quality:change', this.state.quality === 'High' ? 'Low' : 'High');
    });

    this.root.querySelector('#hud-shot-btn')!.addEventListener('click', () => {
      this.bus.emit('screenshot:export', undefined);
    });

    this.root.querySelector('#hud-hide-btn')!.addEventListener('click', () => {
      this.bus.emit('hud:toggle', false);
    });
    this.root.querySelector('#hud-restore-btn')!.addEventListener('click', () => {
      this.bus.emit('hud:toggle', true);
    });

    // Narrow-screen drawers (< 1100px)
    this.root.querySelector('#hud-drawer-left-btn')!.addEventListener('click', () => {
      this.leftDockEl.classList.toggle('drawer-open');
      this.rightDockEl.classList.remove('drawer-open');
    });
    this.root.querySelector('#hud-drawer-right-btn')!.addEventListener('click', () => {
      this.rightDockEl.classList.toggle('drawer-open');
      this.leftDockEl.classList.remove('drawer-open');
    });
  }

  public sync(): void {
    this.root.classList.toggle('hud-hidden', !this.state.hudVisible);
    this.refWrapEl.style.display = this.referenceAvailable ? 'flex' : 'none';

    for (const [mode, btn] of this.motionButtons) {
      btn.classList.toggle('active', this.state.motion === mode);
    }
    for (const [cam, btn] of this.cameraButtons) {
      btn.classList.toggle('active', this.state.cameraPreset === cam);
    }
    for (const [preset, btn] of this.presetButtons) {
      btn.classList.toggle('active', this.state.explodePreset === preset);
    }
    for (const [rm, btn] of this.renderModeButtons) {
      btn.classList.toggle('active', this.state.renderMode === rm);
    }

    this.playBtn.textContent = this.state.playing ? '暂停' : '播放';
    this.speedInput.value = this.state.speed.toFixed(1);
    this.speedValEl.textContent = `${this.state.speed.toFixed(1)}x`;

    const pct = Math.round(this.state.explodeSlider * 100);
    this.explodeInput.value = String(pct);
    this.explodeValEl.textContent =
      this.state.explodePreset === 'custom' ? `${pct}%` : `预设 (${pct}%)`;

    this.orthoBtn.classList.toggle('active', this.state.orthographic);
    this.refBtn.classList.toggle('active', this.state.referenceOverlay);
    const refPct = Math.round(this.state.referenceOpacity * 100);
    this.refOpacityInput.value = String(refPct);
    this.refOpacityValEl.textContent = `${refPct}%`;

    this.qualityBtn.textContent = this.state.quality.toUpperCase();
    this.qualityBtn.classList.toggle('active', this.state.quality === 'High');

    const maxT = Math.max(
      this.state.explodeTarget.t1,
      this.state.explodeTarget.t2,
      this.state.explodeTarget.t3,
    );
    const explodeLabel = maxT < 0.01 ? 'ASSEMBLED' : `EXPLODE ${pct}%`;
    const modeTag =
      this.state.renderMode === 'PBR' ? '' : ` · ${this.state.renderMode.toUpperCase()}`;
    this.statusEl.textContent = `${this.state.motion.toUpperCase()} · ${explodeLabel}${modeTag}`;

    this.renderRightDock();
  }

  public updateTelemetry(fps: number): void {
    this.fpsEl.textContent = `${Math.round(fps)} FPS`;
  }
}
