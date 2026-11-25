import * as THREE from 'three';
import { PARTS_CATALOG_MAP } from '../config/partsCatalog.ts';
import type { EventBus } from '../core/EventBus.ts';
import { PART_RAYCAST_LAYER, type RobotAssembler } from './RobotAssembler.ts';

const CLICK_MAX_DIST_PX = 5;

export class InteractionController {
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointerNdc = new THREE.Vector2();
  private readonly tooltipEl: HTMLDivElement;

  private downX = 0;
  private downY = 0;
  private hoveredPartId: string | null = null;
  private isPointerOverCanvas = false;

  private readonly onPointerMove: (e: PointerEvent) => void;
  private readonly onPointerDown: (e: PointerEvent) => void;
  private readonly onPointerUp: (e: PointerEvent) => void;
  private readonly onPointerLeave: () => void;

  constructor(
    private readonly domElement: HTMLElement,
    private readonly getCamera: () => THREE.Camera,
    private readonly assembler: RobotAssembler,
    private readonly bus: EventBus,
    private readonly onHoverChange: (partId: string | null) => void,
  ) {
    // Restrict raycaster exclusively to part meshes (excludes floor, holo ring, guide lines)
    this.raycaster.layers.set(PART_RAYCAST_LAYER);

    this.tooltipEl = document.createElement('div');
    this.tooltipEl.className = 'hud-part-tooltip';
    this.tooltipEl.style.display = 'none';
    document.body.appendChild(this.tooltipEl);

    this.onPointerMove = (e: PointerEvent): void => {
      this.isPointerOverCanvas = true;
      const rect = this.domElement.getBoundingClientRect();
      this.pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      const hitPartId = this.pickVisiblePart();
      if (hitPartId !== this.hoveredPartId) {
        this.hoveredPartId = hitPartId;
        this.onHoverChange(hitPartId);
        this.domElement.style.cursor = hitPartId ? 'pointer' : 'grab';
      }

      if (hitPartId) {
        const entry = PARTS_CATALOG_MAP.get(hitPartId);
        if (entry) {
          this.tooltipEl.textContent = `${entry.name} · T${entry.tier}`;
          this.tooltipEl.style.display = 'block';
          this.tooltipEl.style.left = `${e.clientX + 14}px`;
          this.tooltipEl.style.top = `${e.clientY - 12}px`;
        }
      } else {
        this.tooltipEl.style.display = 'none';
      }
    };

    this.onPointerDown = (e: PointerEvent): void => {
      this.downX = e.clientX;
      this.downY = e.clientY;
    };

    this.onPointerUp = (e: PointerEvent): void => {
      if (e.button !== 0) return;
      const dist = Math.hypot(e.clientX - this.downX, e.clientY - this.downY);
      if (dist > CLICK_MAX_DIST_PX) return;

      const rect = this.domElement.getBoundingClientRect();
      this.pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      const hitPartId = this.pickVisiblePart();
      this.bus.emit('part:select', hitPartId);
    };

    this.onPointerLeave = (): void => {
      this.isPointerOverCanvas = false;
      this.tooltipEl.style.display = 'none';
      if (this.hoveredPartId !== null) {
        this.hoveredPartId = null;
        this.onHoverChange(null);
      }
    };

    this.domElement.addEventListener('pointermove', this.onPointerMove);
    this.domElement.addEventListener('pointerdown', this.onPointerDown);
    this.domElement.addEventListener('pointerup', this.onPointerUp);
    this.domElement.addEventListener('pointerleave', this.onPointerLeave);
  }

  private pickVisiblePart(): string | null {
    if (!this.isPointerOverCanvas) return null;
    this.raycaster.setFromCamera(this.pointerNdc, this.getCamera());
    const hits = this.raycaster.intersectObjects(this.assembler.allPartMeshes, false);

    for (let i = 0; i < hits.length; i++) {
      const mesh = hits[i]!.object;
      const partId = mesh.userData.partId as string | undefined;
      if (!partId) continue;
      const assembled = this.assembler.getPart(partId);
      if (!assembled || !assembled.container.visible) continue;
      return partId;
    }
    return null;
  }

  public dispose(): void {
    this.domElement.removeEventListener('pointermove', this.onPointerMove);
    this.domElement.removeEventListener('pointerdown', this.onPointerDown);
    this.domElement.removeEventListener('pointerup', this.onPointerUp);
    this.domElement.removeEventListener('pointerleave', this.onPointerLeave);
    this.tooltipEl.remove();
  }
}
