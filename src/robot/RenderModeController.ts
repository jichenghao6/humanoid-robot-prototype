import type { MaterialBank } from '../config/materials.ts';
import type { PartTier } from '../config/partsCatalog.ts';
import type { RenderMode } from '../core/AppState.ts';
import type { AssembledPart } from './RobotAssembler.ts';

/**
 * Manages PBR / X-Ray / Wireframe render modes, hover/selection material highlight,
 * and part visibility (hide / isolate / restoreAll).
 *
 * Strictly zero runtime material allocations: only swaps cached material references.
 */
export class RenderModeController {
  private mode: RenderMode = 'PBR';
  private hoveredPartId: string | null = null;
  private selectedPartId: string | null = null;

  constructor(
    private readonly parts: readonly AssembledPart[],
    private readonly materials: MaterialBank,
  ) {}

  public setMode(mode: RenderMode): void {
    if (this.mode === mode) return;
    this.mode = mode;
    this.refreshAllMaterials();
  }

  public getMode(): RenderMode {
    return this.mode;
  }

  public setHoveredPart(partId: string | null): void {
    if (this.hoveredPartId === partId) return;
    const prev = this.hoveredPartId;
    this.hoveredPartId = partId;
    if (prev) this.refreshPartById(prev);
    if (partId) this.refreshPartById(partId);
  }

  public setSelectedPart(partId: string | null): void {
    if (this.selectedPartId === partId) return;
    const prev = this.selectedPartId;
    this.selectedPartId = partId;
    if (prev) this.refreshPartById(prev);
    if (partId) this.refreshPartById(partId);
  }

  public applyVisibility(hiddenPartIds: ReadonlySet<string>): void {
    for (let i = 0; i < this.parts.length; i++) {
      const part = this.parts[i]!;
      part.container.visible = !hiddenPartIds.has(part.entry.partId);
    }
  }

  public refreshAllMaterials(): void {
    for (let i = 0; i < this.parts.length; i++) {
      this.applyToPart(this.parts[i]!);
    }
  }

  private refreshPartById(partId: string): void {
    for (let i = 0; i < this.parts.length; i++) {
      const part = this.parts[i]!;
      if (part.entry.partId === partId) {
        this.applyToPart(part);
        return;
      }
    }
  }

  private applyToPart(part: AssembledPart): void {
    const isHighlighted =
      part.entry.partId === this.hoveredPartId || part.entry.partId === this.selectedPartId;
    const tier: PartTier = part.entry.tier;

    for (let i = 0; i < part.meshes.length; i++) {
      const mesh = part.meshes[i]!;

      // Grade E LEDs always keep glowing across all render modes (Section 4.4)
      if (mesh.userData.isLed) {
        mesh.material = this.materials.grades.E;
        continue;
      }

      if (this.mode === 'Wireframe') {
        mesh.material = this.materials.wireframe[tier];
        continue;
      }

      const gradeKey = (mesh.userData.materialGrade as string | undefined) ?? 'A';
      if (
        this.mode === 'X-Ray' &&
        (tier === 1 || (tier === 2 && (gradeKey === 'A' || gradeKey === 'B')))
      ) {
        mesh.material = this.materials.xrayShell;
        continue;
      }

      mesh.material = isHighlighted
        ? this.materials.getHighlightMaterial(gradeKey)
        : this.materials.getBaseMaterial(gradeKey);
    }
  }
}
