import type { Side } from './rig.ts';

export type PartTier = 1 | 2 | 3;

export type MaterialGrade = 'A' | 'B' | 'C' | 'D' | 'E' | 'F';

export interface PartCatalogEntry {
  partId: string;
  baseId: string;
  side: Side | null;
  name: string;
  parentJoint: string;
  tier: PartTier;
  explodeVector: readonly [number, number, number];
  materials: readonly MaterialGrade[];
  massKg: number;
  description: string;
}

interface AxialPartSpec {
  partId: string;
  name: string;
  parentJoint: string;
  tier: PartTier;
  explodeVector: readonly [number, number, number];
  materials: readonly MaterialGrade[];
  massKg: number;
  description: string;
}

interface PairedPartSpec {
  baseId: string;
  name: string;
  /** Either an axial joint ('head', 'spine_upper', 'pelvis') or a paired joint prefix ('shoulder', 'pauldron', 'elbow', 'wrist', 'hip', 'patella', 'knee', 'ankle'). */
  parentJointBase: string;
  parentIsPaired: boolean;
  tier: PartTier;
  /** dx > 0 points outward (+X for L, -X for R). */
  explodeVectorOutward: readonly [number, number, number];
  materials: readonly MaterialGrade[];
  massKg: number;
  description: string;
}

const AXIAL_SPECS: readonly AxialPartSpec[] = [
  {
    partId: 'head_helmet_shell',
    name: '头部流线外壳',
    parentJoint: 'head',
    tier: 1,
    explodeVector: [0.00, 0.22, -0.06],
    materials: ['A'],
    massKg: 1.1,
    description: '航空陶瓷复合外壳，保护顶部视觉与通信阵列',
  },
  {
    partId: 'head_visor',
    name: '曜石黑光学面罩（含 V 型眼眶灯带）',
    parentJoint: 'head',
    tier: 1,
    explodeVector: [0.00, 0.06, 0.24],
    materials: ['D', 'E'],
    massKg: 0.6,
    description: '多光谱透波面罩，集成状态光导',
  },
  {
    partId: 'head_ai_core',
    name: '头部 AI 推理模组',
    parentJoint: 'head',
    tier: 3,
    explodeVector: [0.00, 0.10, 0.00],
    materials: ['F'],
    massKg: 0.8,
    description: '端侧推理模组，带铜铝散热鳍片',
  },
  {
    partId: 'neck_actuator',
    name: '多连杆颈部致动柱',
    parentJoint: 'neck',
    tier: 2,
    explodeVector: [0.00, 0.05, -0.04],
    materials: ['C'],
    massKg: 1.2,
    description: '并联柔性机构，负责俯仰与偏航',
  },
  {
    partId: 'chest_front_armor',
    name: '胸甲与倒三角能量核',
    parentJoint: 'spine_upper',
    tier: 1,
    explodeVector: [0.00, 0.06, 0.30],
    materials: ['A', 'B', 'E'],
    massKg: 3.2,
    description: '胸部主防护甲，中央为状态指示核心',
  },
  {
    partId: 'spine_energy_bar',
    name: '背部竖向脊柱能量核',
    parentJoint: 'spine_upper',
    tier: 2,
    explodeVector: [0.00, 0.04, -0.20],
    materials: ['C', 'E'],
    massKg: 1.0,
    description: '高密度能量传输母线',
  },
  {
    partId: 'torso_battery_pack',
    name: '胸腔固态电池包',
    parentJoint: 'spine_upper',
    tier: 3,
    explodeVector: [0.00, 0.00, 0.10],
    materials: ['F'],
    massKg: 10.0,
    description: '2.3 kWh 固态电池，经背部检修口快拆',
  },
  {
    partId: 'torso_compute_board',
    name: '主控板与铜质热管',
    parentJoint: 'spine_upper',
    tier: 3,
    explodeVector: [0.00, 0.10, -0.08],
    materials: ['F'],
    massKg: 1.6,
    description: '全身运动控制主板，铜热管导热至背部',
  },
  {
    partId: 'torso_main_chassis',
    name: '胸腔钛合金主框架',
    parentJoint: 'spine_upper',
    tier: 3,
    explodeVector: [0.00, 0.00, 0.00],
    materials: ['C'],
    massKg: 5.0,
    description: '拓扑优化承力框架（锚点，不移动）',
  },
  {
    partId: 'abdomen_hex_plate',
    name: '六边形腹板与两侧弧形光带',
    parentJoint: 'spine_lower',
    tier: 1,
    explodeVector: [0.00, -0.02, 0.18],
    materials: ['C', 'E'],
    massKg: 1.3,
    description: '三节仿生伸缩护腹，两侧为散热光槽',
  },
  {
    partId: 'lumbar_actuator',
    name: '腰部三轴驱动与稳定连杆',
    parentJoint: 'spine_lower',
    tier: 2,
    explodeVector: [0.00, -0.02, -0.14],
    materials: ['C'],
    massKg: 2.4,
    description: '双轴液压阻尼，吸收落地冲击',
  },
  {
    partId: 'pelvis_front_armor',
    name: '骨盆前部 V 型护甲',
    parentJoint: 'pelvis',
    tier: 1,
    explodeVector: [0.00, -0.04, 0.22],
    materials: ['A'],
    massKg: 1.6,
    description: '保护髋部总线与 IMU',
  },
  {
    partId: 'pelvis_gimbal_base',
    name: '骨盆核心与 IMU 底座',
    parentJoint: 'pelvis',
    tier: 3,
    explodeVector: [0.00, 0.00, 0.00],
    materials: ['C', 'F'],
    massKg: 3.5,
    description: '全身动力学根基准（锚点，不移动）',
  },
];

const PAIRED_SPECS: readonly PairedPartSpec[] = [
  {
    baseId: 'head_ear_pod',
    name: '耳部传感器舱（含蓝光环）',
    parentJointBase: 'head',
    parentIsPaired: false,
    tier: 2,
    explodeVectorOutward: [0.18, 0.03, 0.00],
    materials: ['C', 'E'],
    massKg: 0.25,
    description: '空间拾音与毫米波雷达',
  },
  {
    baseId: 'back_scapula_armor',
    name: '背部翼状肩胛甲',
    parentJointBase: 'spine_upper',
    parentIsPaired: false,
    tier: 1,
    explodeVectorOutward: [0.16, 0.06, -0.26],
    materials: ['A'],
    massKg: 1.4,
    description: '仿生导流护板，兼顾散热',
  },
  {
    baseId: 'pelvis_rear_armor',
    name: '骨盆后部分片护甲',
    parentJointBase: 'pelvis',
    parentIsPaired: false,
    tier: 1,
    explodeVectorOutward: [0.10, -0.04, -0.20],
    materials: ['A'],
    massKg: 0.6,
    description: '预留髋部大角度后伸行程',
  },
  {
    baseId: 'hip_actuator',
    name: '髋部外露圆形致动器',
    parentJointBase: 'pelvis',
    parentIsPaired: false,
    tier: 2,
    explodeVectorOutward: [0.16, 0.00, 0.00],
    materials: ['C'],
    massKg: 1.8,
    description: '三轴髋驱动，峰值 220 N·m',
  },
  {
    baseId: 'shoulder_pauldron',
    name: '浮动三角肌肩甲',
    parentJointBase: 'pauldron',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.26, 0.14, 0.00],
    materials: ['A', 'B', 'E'],
    massKg: 0.9,
    description: '悬浮式白甲，内侧带蓝色光导',
  },
  {
    baseId: 'shoulder_actuator',
    name: '肩部三轴谐波驱动器',
    parentJointBase: 'shoulder',
    parentIsPaired: true,
    tier: 2,
    explodeVectorOutward: [0.14, 0.05, 0.00],
    materials: ['C'],
    massKg: 1.5,
    description: '峰值 120 N·m',
  },
  {
    baseId: 'upper_arm_armor',
    name: '上臂白甲套筒',
    parentJointBase: 'shoulder',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.22, 0.00, 0.04],
    materials: ['A'],
    massKg: 0.7,
    description: '双片式轻量化外壳',
  },
  {
    baseId: 'humerus_frame',
    name: '上臂承力杆与线束',
    parentJointBase: 'shoulder',
    parentIsPaired: true,
    tier: 3,
    explodeVectorOutward: [0.06, 0.00, -0.03],
    materials: ['C'],
    massKg: 0.9,
    description: '中空钛合金承力杆',
  },
  {
    baseId: 'elbow_actuator',
    name: '肘部同心圆致动器',
    parentJointBase: 'elbow',
    parentIsPaired: true,
    tier: 2,
    explodeVectorOutward: [0.14, 0.00, -0.06],
    materials: ['C'],
    massKg: 0.8,
    description: '单轴铰链，屈曲 0°~135°',
  },
  {
    baseId: 'forearm_gauntlet',
    name: '前臂护甲（含斜向灯带）',
    parentJointBase: 'elbow',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.22, -0.04, 0.08],
    materials: ['A', 'E'],
    massKg: 0.6,
    description: '集成腕部力矩传感总线',
  },
  {
    baseId: 'forearm_frame',
    name: '前臂骨架与腕部驱动',
    parentJointBase: 'elbow',
    parentIsPaired: true,
    tier: 3,
    explodeVectorOutward: [0.06, -0.02, -0.03],
    materials: ['C'],
    massKg: 0.7,
    description: '双连杆腕部驱动',
  },
  {
    baseId: 'hand_assembly',
    name: '五指灵巧手（整体）',
    parentJointBase: 'wrist',
    parentIsPaired: true,
    tier: 2,
    explodeVectorOutward: [0.18, -0.14, 0.06],
    materials: ['A', 'C'],
    massKg: 0.9,
    description: '五指独立驱动，带触觉指尖',
  },
  {
    baseId: 'thigh_front_armor',
    name: '大腿前外侧主装甲',
    parentJointBase: 'hip',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.22, -0.02, 0.18],
    materials: ['A', 'E'],
    massKg: 1.8,
    description: '内嵌长条形电光蓝光槽',
  },
  {
    baseId: 'thigh_rear_armor',
    name: '大腿后侧散热护甲',
    parentJointBase: 'hip',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.16, -0.02, -0.18],
    materials: ['A', 'E'],
    massKg: 1.0,
    description: '主动散热百叶与竖向灯条',
  },
  {
    baseId: 'femur_frame',
    name: '股骨骨架与直线推杆',
    parentJointBase: 'hip',
    parentIsPaired: true,
    tier: 3,
    explodeVectorOutward: [0.08, 0.00, 0.00],
    materials: ['C'],
    massKg: 2.0,
    description: '中空钛合金承重骨架',
  },
  {
    baseId: 'knee_patella_guard',
    name: '浮动护膝',
    parentJointBase: 'patella',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.06, 0.02, 0.24],
    materials: ['A'],
    massKg: 0.4,
    description: '随膝角滑动避让',
  },
  {
    baseId: 'knee_drum_actuator',
    name: '膝部滚筒致动器',
    parentJointBase: 'knee',
    parentIsPaired: true,
    tier: 2,
    explodeVectorOutward: [0.12, 0.00, 0.00],
    materials: ['C'],
    massKg: 2.0,
    description: '峰值 280 N·m',
  },
  {
    baseId: 'shin_front_armor',
    name: '小腿前护胫（含分段灯条）',
    parentJointBase: 'knee',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.18, -0.04, 0.16],
    materials: ['A', 'E'],
    massKg: 1.1,
    description: '流线护胫',
  },
  {
    baseId: 'calf_rear_armor',
    name: '小腿后腿肚甲（含跟腱灯条）',
    parentJointBase: 'knee',
    parentIsPaired: true,
    tier: 1,
    explodeVectorOutward: [0.18, -0.04, -0.16],
    materials: ['A', 'E'],
    massKg: 0.8,
    description: '分片腿肚甲',
  },
  {
    baseId: 'tibia_frame',
    name: '胫骨骨架与踝部推杆',
    parentJointBase: 'knee',
    parentIsPaired: true,
    tier: 3,
    explodeVectorOutward: [0.06, 0.00, 0.00],
    materials: ['C'],
    massKg: 1.4,
    description: '双推杆驱动踝关节',
  },
  {
    baseId: 'ankle_actuator',
    name: '踝部二轴致动器（含光环）',
    parentJointBase: 'ankle',
    parentIsPaired: true,
    tier: 2,
    explodeVectorOutward: [0.12, 0.00, -0.04],
    materials: ['C', 'E'],
    massKg: 0.9,
    description: '六维力传感器集成',
  },
  {
    baseId: 'foot_assembly',
    name: '分体机械足（整体）',
    parentJointBase: 'ankle',
    parentIsPaired: true,
    tier: 2,
    explodeVectorOutward: [0.10, 0.00, 0.06],
    materials: ['A', 'C'],
    massKg: 1.6,
    description: '前掌弯折铰链 + 高阻尼鞋底',
  },
];

function buildPartsCatalog(): PartCatalogEntry[] {
  const entries: PartCatalogEntry[] = [];

  for (const spec of AXIAL_SPECS) {
    entries.push({
      partId: spec.partId,
      baseId: spec.partId,
      side: null,
      name: spec.name,
      parentJoint: spec.parentJoint,
      tier: spec.tier,
      explodeVector: spec.explodeVector,
      materials: spec.materials,
      massKg: spec.massKg,
      description: spec.description,
    });
  }

  const sides: readonly Side[] = ['L', 'R'];
  for (const spec of PAIRED_SPECS) {
    for (const side of sides) {
      const signX = side === 'L' ? 1 : -1;
      const sideLabel = side === 'L' ? '（左）' : '（右）';
      entries.push({
        partId: `${spec.baseId}_${side}`,
        baseId: spec.baseId,
        side,
        name: `${spec.name}${sideLabel}`,
        parentJoint: spec.parentIsPaired ? `${spec.parentJointBase}_${side}` : spec.parentJointBase,
        tier: spec.tier,
        explodeVector: [
          spec.explodeVectorOutward[0] * signX,
          spec.explodeVectorOutward[1],
          spec.explodeVectorOutward[2],
        ],
        materials: spec.materials,
        massKg: spec.massKg,
        description: spec.description,
      });
    }
  }

  return entries;
}

export const PARTS_CATALOG: readonly PartCatalogEntry[] = buildPartsCatalog();

export const PARTS_CATALOG_MAP: ReadonlyMap<string, PartCatalogEntry> = new Map(
  PARTS_CATALOG.map((p) => [p.partId, p]),
);

export interface CatalogStats {
  entryTypes: number;
  totalParts: number;
  totalMassKg: number;
  tierCounts: Record<PartTier, number>;
}

export function computeCatalogStats(): CatalogStats {
  const tierCounts: Record<PartTier, number> = { 1: 0, 2: 0, 3: 0 };
  let totalMassKg = 0;

  for (const part of PARTS_CATALOG) {
    tierCounts[part.tier] += 1;
    totalMassKg += part.massKg;
  }

  return {
    entryTypes: AXIAL_SPECS.length + PAIRED_SPECS.length,
    totalParts: PARTS_CATALOG.length,
    totalMassKg: Math.round(totalMassKg * 100) / 100,
    tierCounts,
  };
}
