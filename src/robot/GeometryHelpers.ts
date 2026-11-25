import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

export interface CurvedArmorPlateOptions {
  thickness: number;
  bevel: number;
  /** Cylindrical curvature radius around Y axis (bends X into Z). Positive curves edges toward -Z. */
  bendRadiusX?: number;
  /** Cylindrical curvature radius around X axis (bends Y into Z). Positive curves edges toward -Z. */
  bendRadiusY?: number;
  /** Subdivision grid count along U and V (default 10). */
  segments?: number;
  /** Optional smooth outward dome bulge at center (meters). */
  crownZ?: number;
}

/**
 * Normalizes a BufferGeometry to non-indexed with position, normal, and uv attributes
 * so mergeGeometries never fails due to attribute mismatch.
 */
export function normalizeGeometryForMerge(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  const g = geo.index ? geo.toNonIndexed() : geo.clone();
  if (!g.getAttribute('normal')) {
    g.computeVertexNormals();
  }
  const posAttr = g.getAttribute('position');
  if (!g.getAttribute('uv')) {
    const uvs = new Float32Array(posAttr.count * 2);
    for (let i = 0; i < posAttr.count; i++) {
      uvs[i * 2] = posAttr.getX(i) * 4 + 0.5;
      uvs[i * 2 + 1] = posAttr.getY(i) * 4 + 0.5;
    }
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  }
  // Remove tangent or color attributes if inconsistent across parts
  for (const key of Object.keys(g.attributes)) {
    if (key !== 'position' && key !== 'normal' && key !== 'uv') {
      g.deleteAttribute(key);
    }
  }
  return g;
}

/**
 * Safely merges an array of BufferGeometries and disposes the inputs.
 */
export function mergeGeometriesSafe(
  geometries: THREE.BufferGeometry[],
  disposeInputs = true,
): THREE.BufferGeometry {
  if (geometries.length === 0) {
    return new THREE.BoxGeometry(0.001, 0.001, 0.001);
  }
  const normalized = geometries.map((g) => normalizeGeometryForMerge(g));
  const merged = mergeGeometries(normalized, false);
  for (const n of normalized) n.dispose();
  if (disposeInputs) {
    for (const g of geometries) g.dispose();
  }
  if (!merged) {
    throw new Error('mergeGeometriesSafe: failed to merge geometries');
  }
  return merged;
}

/**
 * Computes area-weighted vertex normals with hard-surface crease splitting:
 * adjacent faces whose dihedral angle is <= creaseAngleRad blend smoothly,
 * while sharper chamfer/cap edges (> creaseAngleRad) remain crisp.
 */
export function applyCreasedNormals(
  geo: THREE.BufferGeometry,
  creaseAngleRad = Math.PI * 0.21,
): THREE.BufferGeometry {
  const nonIndexed = geo.index ? geo.toNonIndexed() : geo;
  const pos = nonIndexed.getAttribute('position');
  const vCount = pos.count;
  const fCount = Math.floor(vCount / 3);
  if (fCount === 0) return nonIndexed;

  const faceAreaNormals = new Float32Array(fCount * 3);
  const faceUnitNormals = new Float32Array(fCount * 3);
  const buckets = new Map<string, number[]>();
  const quantScale = 1e5; // 0.01 mm precision so 3-4 mm bevels never weld across ridges

  for (let f = 0; f < fCount; f++) {
    const i0 = f * 3;
    const ax = pos.getX(i0);
    const ay = pos.getY(i0);
    const az = pos.getZ(i0);
    const bx = pos.getX(i0 + 1);
    const by = pos.getY(i0 + 1);
    const bz = pos.getZ(i0 + 1);
    const cx = pos.getX(i0 + 2);
    const cy = pos.getY(i0 + 2);
    const cz = pos.getZ(i0 + 2);

    const abx = bx - ax;
    const aby = by - ay;
    const abz = bz - az;
    const acx = cx - ax;
    const acy = cy - ay;
    const acz = cz - az;

    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const len = Math.hypot(nx, ny, nz);

    const f3 = f * 3;
    faceAreaNormals[f3] = nx;
    faceAreaNormals[f3 + 1] = ny;
    faceAreaNormals[f3 + 2] = nz;
    if (len > 1e-12) {
      faceUnitNormals[f3] = nx / len;
      faceUnitNormals[f3 + 1] = ny / len;
      faceUnitNormals[f3 + 2] = nz / len;
    }

    for (let k = 0; k < 3; k++) {
      const vi = i0 + k;
      const key = `${Math.round(pos.getX(vi) * quantScale)},${Math.round(pos.getY(vi) * quantScale)},${Math.round(pos.getZ(vi) * quantScale)}`;
      let list = buckets.get(key);
      if (!list) {
        list = [];
        buckets.set(key, list);
      }
      list.push(f);
    }
  }

  const creaseDot = Math.cos(creaseAngleRad);
  const normals = new Float32Array(vCount * 3);

  for (let f = 0; f < fCount; f++) {
    const f3 = f * 3;
    const fnx = faceUnitNormals[f3]!;
    const fny = faceUnitNormals[f3 + 1]!;
    const fnz = faceUnitNormals[f3 + 2]!;
    const hasUnit = fnx * fnx + fny * fny + fnz * fnz > 0.5;

    for (let k = 0; k < 3; k++) {
      const vi = f * 3 + k;
      const key = `${Math.round(pos.getX(vi) * quantScale)},${Math.round(pos.getY(vi) * quantScale)},${Math.round(pos.getZ(vi) * quantScale)}`;
      const faces = buckets.get(key)!;

      let sx = 0;
      let sy = 0;
      let sz = 0;
      for (let j = 0; j < faces.length; j++) {
        const f2 = faces[j]!;
        const f23 = f2 * 3;
        const u2x = faceUnitNormals[f23]!;
        const u2y = faceUnitNormals[f23 + 1]!;
        const u2z = faceUnitNormals[f23 + 2]!;
        if (!hasUnit || fnx * u2x + fny * u2y + fnz * u2z >= creaseDot) {
          sx += faceAreaNormals[f23]!;
          sy += faceAreaNormals[f23 + 1]!;
          sz += faceAreaNormals[f23 + 2]!;
        }
      }

      const sLen = Math.hypot(sx, sy, sz);
      const v3 = vi * 3;
      if (sLen > 1e-12) {
        normals[v3] = sx / sLen;
        normals[v3 + 1] = sy / sLen;
        normals[v3 + 2] = sz / sLen;
      } else if (hasUnit) {
        normals[v3] = fnx;
        normals[v3 + 1] = fny;
        normals[v3 + 2] = fnz;
      } else {
        normals[v3] = 0;
        normals[v3 + 1] = 1;
        normals[v3 + 2] = 0;
      }
    }
  }

  nonIndexed.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return nonIndexed;
}

/**
 * Geometry-level X-axis mirror (Section 3.1.4):
 * Negates X coordinates, reverses triangle winding order, and preserves smooth/creased normals.
 * Never uses negative Object3D scale.
 */
export function mirrorGeometryX(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  const mirrored = geo.clone();
  mirrored.scale(-1, 1, 1);

  const nrm = mirrored.getAttribute('normal');
  if (mirrored.index) {
    const arr = mirrored.index.array as Uint16Array | Uint32Array;
    for (let i = 0; i < arr.length; i += 3) {
      const tmp = arr[i + 1]!;
      arr[i + 1] = arr[i + 2]!;
      arr[i + 2] = tmp;
    }
    mirrored.index.needsUpdate = true;
    if (nrm) {
      for (let i = 0; i < nrm.count; i++) {
        nrm.setX(i, -nrm.getX(i));
      }
      nrm.needsUpdate = true;
    } else {
      mirrored.computeVertexNormals();
    }
  } else {
    const pos = mirrored.getAttribute('position');
    const uv = mirrored.getAttribute('uv');
    for (let i = 0; i < pos.count; i += 3) {
      const x1 = pos.getX(i + 1);
      const y1 = pos.getY(i + 1);
      const z1 = pos.getZ(i + 1);
      pos.setXYZ(i + 1, pos.getX(i + 2), pos.getY(i + 2), pos.getZ(i + 2));
      pos.setXYZ(i + 2, x1, y1, z1);

      if (uv) {
        const u1 = uv.getX(i + 1);
        const v1 = uv.getY(i + 1);
        uv.setXY(i + 1, uv.getX(i + 2), uv.getY(i + 2));
        uv.setXY(i + 2, u1, v1);
      }

      if (nrm) {
        const nx1 = nrm.getX(i + 1);
        const ny1 = nrm.getY(i + 1);
        const nz1 = nrm.getZ(i + 1);
        nrm.setXYZ(i + 1, -nrm.getX(i + 2), nrm.getY(i + 2), nrm.getZ(i + 2));
        nrm.setXYZ(i + 2, -nx1, ny1, nz1);
        nrm.setX(i, -nrm.getX(i));
      }
    }
    pos.needsUpdate = true;
    if (uv) uv.needsUpdate = true;
    if (nrm) nrm.needsUpdate = true;
    else mirrored.computeVertexNormals();
  }

  return mirrored;
}

/**
 * Evaluates a curved surface mapping (x, y) -> (pos, normal) with optional cylindrical/spherical bend & crown.
 */
function mapPointToCurvedSurface(
  x: number,
  y: number,
  uNorm: number,
  vNorm: number,
  opts: CurvedArmorPlateOptions,
  outPos: THREE.Vector3,
  outNormal: THREE.Vector3,
): void {
  let px = x;
  let py = y;
  let pz = 0;
  let nx = 0;
  let ny = 0;
  let nz = 1;

  if (opts.bendRadiusX && Math.abs(opts.bendRadiusX) > 1e-4) {
    const r = opts.bendRadiusX;
    const theta = x / r;
    px = r * Math.sin(theta);
    pz -= r * (1 - Math.cos(theta));
    nx += Math.sin(theta);
    nz *= Math.cos(theta);
  }

  if (opts.bendRadiusY && Math.abs(opts.bendRadiusY) > 1e-4) {
    const r = opts.bendRadiusY;
    const phi = y / r;
    py = r * Math.sin(phi);
    pz -= r * (1 - Math.cos(phi));
    ny += Math.sin(phi);
    nz *= Math.cos(phi);
  }

  if (opts.crownZ && Math.abs(opts.crownZ) > 1e-5) {
    const dome = Math.sin(Math.PI * uNorm) * Math.sin(Math.PI * vNorm);
    pz += opts.crownZ * dome;
  }

  outPos.set(px, py, pz);
  outNormal.set(nx, ny, nz).normalize();
}

/**
 * Hard-surface Curved Armor Plate Generator (Section 3.1.1):
 * Builds a clean star-convex subdivided radial/grid shell from a 2D outline polygon (CCW),
 * maps all vertices onto a cylindrical/spherical curved surface, offsets along normals for thickness,
 * and adds crisp chamfered bevel bands along the entire perimeter — zero ExtrudeGeometry creasing!
 */
export function createCurvedArmorPlate(
  outline: readonly THREE.Vector2[],
  opts: CurvedArmorPlateOptions,
): THREE.BufferGeometry {
  const segs = Math.max(4, opts.segments ?? 8);
  const thick = Math.max(0.002, opts.thickness);
  const bevel = Math.min(thick * 0.45, Math.max(0.0005, opts.bevel));

  // Ensure 2D outline is strictly CCW (positive signed area) so front/side normals always point outward
  let signedArea2 = 0;
  for (let i = 0; i < outline.length; i++) {
    const a = outline[i]!;
    const b = outline[(i + 1) % outline.length]!;
    signedArea2 += a.x * b.y - b.x * a.y;
  }
  const ccwOutline = signedArea2 < 0 ? [...outline].reverse() : outline;

  // Compute centroid & bounding box of 2D outline
  let cx = 0;
  let cy = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const pt of ccwOutline) {
    cx += pt.x;
    cy += pt.y;
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }
  cx /= ccwOutline.length;
  cy /= ccwOutline.length;
  const spanX = Math.max(1e-4, maxX - minX);
  const spanY = Math.max(1e-4, maxY - minY);

  // Resample perimeter evenly along polygon edges for dense curved side walls
  const perim: THREE.Vector2[] = [];
  const edgeSubs = Math.max(2, Math.round(segs / 3));
  for (let i = 0; i < ccwOutline.length; i++) {
    const a = ccwOutline[i]!;
    const b = ccwOutline[(i + 1) % ccwOutline.length]!;
    for (let s = 0; s < edgeSubs; s++) {
      const t = s / edgeSubs;
      perim.push(new THREE.Vector2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t));
    }
  }
  const nPerim = perim.length;

  // Build radial rings from center (ring 0) to bevel-inset ring (rings-1) to outer edge (rings)
  const rings = segs;
  const bevelFrac = 1 - Math.min(0.18, bevel / (Math.min(spanX, spanY) * 0.5));

  const positions: number[] = [];
  const uvs: number[] = [];
  const pos = new THREE.Vector3();
  const nrm = new THREE.Vector3();

  const evalSurface = (
    x: number,
    y: number,
    normalOffset: number,
    out: THREE.Vector3,
  ): void => {
    const u = (x - minX) / spanX;
    const v = (y - minY) / spanY;
    mapPointToCurvedSurface(x, y, u, v, opts, pos, nrm);
    out.copy(pos).addScaledVector(nrm, normalOffset);
  };

  // Precompute grid rings for front top face, front bevel edge, back bevel edge, back face
  const frontGrid: THREE.Vector3[][] = [];
  const backGrid: THREE.Vector3[][] = [];

  for (let r = 0; r <= rings; r++) {
    const frac = (r / rings) * bevelFrac;
    const fRing: THREE.Vector3[] = [];
    const bRing: THREE.Vector3[] = [];
    for (let i = 0; i < nPerim; i++) {
      const p = perim[i]!;
      const x = cx + (p.x - cx) * frac;
      const y = cy + (p.y - cy) * frac;
      const fp = new THREE.Vector3();
      const bp = new THREE.Vector3();
      evalSurface(x, y, thick * 0.5, fp);
      evalSurface(x, y, -thick * 0.5, bp);
      fRing.push(fp);
      bRing.push(bp);
    }
    frontGrid.push(fRing);
    backGrid.push(bRing);
  }

  // Outer perimeter rim at chamfered height (frontRim & backRim)
  const frontRim: THREE.Vector3[] = [];
  const backRim: THREE.Vector3[] = [];
  for (let i = 0; i < nPerim; i++) {
    const p = perim[i]!;
    const fr = new THREE.Vector3();
    const br = new THREE.Vector3();
    evalSurface(p.x, p.y, thick * 0.5 - bevel, fr);
    evalSurface(p.x, p.y, -thick * 0.5 + bevel, br);
    frontRim.push(fr);
    backRim.push(br);
  }

  const pushTri = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3): void => {
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    uvs.push(
      (a.x - minX) / spanX,
      (a.y - minY) / spanY,
      (b.x - minX) / spanX,
      (b.y - minY) / spanY,
      (c.x - minX) / spanX,
      (c.y - minY) / spanY,
    );
  };

  const pushQuad = (
    a: THREE.Vector3,
    b: THREE.Vector3,
    c: THREE.Vector3,
    d: THREE.Vector3,
  ): void => {
    pushTri(a, b, c);
    pushTri(a, c, d);
  };

  // Front face rings (CCW outward)
  for (let r = 0; r < rings; r++) {
    const r0 = frontGrid[r]!;
    const r1 = frontGrid[r + 1]!;
    for (let i = 0; i < nPerim; i++) {
      const next = (i + 1) % nPerim;
      if (r === 0) {
        pushTri(r0[0]!, r1[i]!, r1[next]!);
      } else {
        pushQuad(r0[i]!, r1[i]!, r1[next]!, r0[next]!);
      }
    }
  }

  // Back face rings (reversed winding)
  for (let r = 0; r < rings; r++) {
    const r0 = backGrid[r]!;
    const r1 = backGrid[r + 1]!;
    for (let i = 0; i < nPerim; i++) {
      const next = (i + 1) % nPerim;
      if (r === 0) {
        pushTri(r0[0]!, r1[next]!, r1[i]!);
      } else {
        pushQuad(r0[i]!, r0[next]!, r1[next]!, r1[i]!);
      }
    }
  }

  // Front bevel strip, side wall strip, back bevel strip (split faces give crisp specular chamfers!)
  const fOuter = frontGrid[rings]!;
  const bOuter = backGrid[rings]!;
  for (let i = 0; i < nPerim; i++) {
    const next = (i + 1) % nPerim;
    pushQuad(fOuter[i]!, frontRim[i]!, frontRim[next]!, fOuter[next]!);
    pushQuad(frontRim[i]!, backRim[i]!, backRim[next]!, frontRim[next]!);
    pushQuad(backRim[i]!, bOuter[i]!, bOuter[next]!, backRim[next]!);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  return applyCreasedNormals(geo, Math.PI * 0.21);
}

export interface LoftStation {
  y: number;
  centerX: number;
  centerZ: number;
  rx: number;
  rz: number;
  /** Optional cross-section shape flattening or angularity (0 = ellipse, 0.35 = superellipse armor) */
  squareness?: number;
  /** Start/end angle in radians for open curved shells (default full 0..2PI closed tube) */
  angleStart?: number;
  angleEnd?: number;
}

/**
 * Builds a sleek multi-station lofted solid shell (closed or partial arc with thickness)
 * for sculpted armored limbs (thigh, shin, calf, sneaker upper).
 */
export function createLoftedArmorShell(
  stations: readonly LoftStation[],
  radialSegments = 20,
  thickness = 0.012,
  closedTube = false,
): THREE.BufferGeometry {
  const positions: number[] = [];
  const uvs: number[] = [];

  const evalRing = (st: LoftStation, offsetR: number): THREE.Vector3[] => {
    const pts: THREE.Vector3[] = [];
    const a0 = st.angleStart ?? 0;
    const a1 = st.angleEnd ?? Math.PI * 2;
    const sq = st.squareness ?? 0.25;
    const count = closedTube ? radialSegments : radialSegments + 1;
    for (let i = 0; i < count; i++) {
      const t = closedTube ? i / radialSegments : i / radialSegments;
      const a = a0 + (a1 - a0) * t;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      // Superellipse shaping for crisp hard-surface aerospace contour
      const exp = 1 - sq * 0.42;
      const sx = Math.sign(ca) * Math.pow(Math.abs(ca), exp);
      const sz = Math.sign(sa) * Math.pow(Math.abs(sa), exp);
      const rx = Math.max(0.002, st.rx + offsetR);
      const rz = Math.max(0.002, st.rz + offsetR);
      pts.push(new THREE.Vector3(st.centerX + sx * rx, st.y, st.centerZ + sz * rz));
    }
    return pts;
  };

  const outerRings = stations.map((s) => evalRing(s, 0));
  const innerRings = closedTube ? [] : stations.map((s) => evalRing(s, -thickness));

  const pushTri = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3): void => {
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    uvs.push(a.x * 4, a.y * 4, b.x * 4, b.y * 4, c.x * 4, c.y * 4);
  };

  const pushQuad = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, d: THREE.Vector3): void => {
    pushTri(a, b, c);
    pushTri(a, c, d);
  };

  const nRows = stations.length - 1;
  for (let row = 0; row < nRows; row++) {
    const o0 = outerRings[row]!;
    const o1 = outerRings[row + 1]!;
    const cols = radialSegments;

    for (let col = 0; col < cols; col++) {
      const next = closedTube ? (col + 1) % radialSegments : col + 1;
      // Outer surface (verified outward normal in right-handed Y-up coords)
      pushQuad(o0[col]!, o1[col]!, o1[next]!, o0[next]!);
    }

    if (!closedTube) {
      const i0 = innerRings[row]!;
      const i1 = innerRings[row + 1]!;
      for (let col = 0; col < cols; col++) {
        const next = col + 1;
        // Inner surface (reversed)
        pushQuad(i0[col]!, i0[next]!, i1[next]!, i1[col]!);
      }
      // Start & end longitudinal side walls (outward normals)
      pushQuad(o0[0]!, i0[0]!, i1[0]!, o1[0]!);
      const last = radialSegments;
      pushQuad(o0[last]!, o1[last]!, i1[last]!, i0[last]!);
    }
  }

  // Cap bottom (row 0, normal -Y) and top (row nRows, normal +Y)
  if (closedTube) {
    const bot = outerRings[0]!;
    const top = outerRings[nRows]!;
    const botCenter = new THREE.Vector3(stations[0]!.centerX, stations[0]!.y, stations[0]!.centerZ);
    const topCenter = new THREE.Vector3(
      stations[nRows]!.centerX,
      stations[nRows]!.y,
      stations[nRows]!.centerZ,
    );
    for (let col = 0; col < radialSegments; col++) {
      const next = (col + 1) % radialSegments;
      pushTri(botCenter, bot[col]!, bot[next]!);
      pushTri(topCenter, top[next]!, top[col]!);
    }
  } else {
    const oBot = outerRings[0]!;
    const iBot = innerRings[0]!;
    const oTop = outerRings[nRows]!;
    const iTop = innerRings[nRows]!;
    for (let col = 0; col < radialSegments; col++) {
      const next = col + 1;
      pushQuad(oBot[col]!, oBot[next]!, iBot[next]!, iBot[col]!);
      pushQuad(oTop[col]!, iTop[col]!, iTop[next]!, oTop[next]!);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  return applyCreasedNormals(geo, Math.PI * 0.21);
}

/**
 * Builds a concentric machined actuator drum aligned along the X-axis (standard hinge axis)
 * using LatheGeometry + recessed step grooves + radial bolt ring.
 */
export function createConcentricDrumGeometry(opts: {
  outerRadius: number;
  width: number;
  innerHubRadius?: number;
  boltCount?: number;
  segments?: number;
}): THREE.BufferGeometry {
  const r = opts.outerRadius;
  const hw = opts.width * 0.5;
  const hubR = opts.innerHubRadius ?? r * 0.52;
  const segs = opts.segments ?? 28;
  const parts: THREE.BufferGeometry[] = [];

  // Lathe profile around Y, then rotated onto X-axis
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(0.001, -hw - 0.003),
    new THREE.Vector2(hubR * 0.65, -hw - 0.003),
    new THREE.Vector2(hubR * 0.78, -hw + 0.002),
    new THREE.Vector2(hubR, -hw + 0.002),
    new THREE.Vector2(hubR * 1.08, -hw - 0.002),
    new THREE.Vector2(r * 0.88, -hw - 0.002),
    new THREE.Vector2(r * 0.94, -hw + 0.004),
    new THREE.Vector2(r, -hw + 0.008),
    new THREE.Vector2(r, -hw * 0.35),
    new THREE.Vector2(r * 0.94, -hw * 0.25),
    new THREE.Vector2(r * 0.94, hw * 0.25),
    new THREE.Vector2(r, hw * 0.35),
    new THREE.Vector2(r, hw - 0.008),
    new THREE.Vector2(r * 0.94, hw - 0.004),
    new THREE.Vector2(r * 0.88, hw + 0.002),
    new THREE.Vector2(hubR * 1.08, hw + 0.002),
    new THREE.Vector2(hubR, hw - 0.002),
    new THREE.Vector2(hubR * 0.78, hw - 0.002),
    new THREE.Vector2(hubR * 0.65, hw + 0.003),
    new THREE.Vector2(0.001, hw + 0.003),
  ];

  const mainLathe = new THREE.LatheGeometry(pts, segs);
  mainLathe.rotateZ(Math.PI / 2);
  parts.push(mainLathe);

  // Outer bearing chamfer ring
  const ringL = new THREE.TorusGeometry(r * 0.76, r * 0.055, 8, segs);
  ringL.rotateY(Math.PI / 2);
  ringL.translate(hw + 0.001, 0, 0);
  parts.push(ringL);

  const ringR = new THREE.TorusGeometry(r * 0.76, r * 0.055, 8, segs);
  ringR.rotateY(Math.PI / 2);
  ringR.translate(-hw - 0.001, 0, 0);
  parts.push(ringR);

  // Radial socket bolts on both faces
  const boltCount = opts.boltCount ?? 6;
  const boltRingR = r * 0.62;
  const boltRadius = r * 0.065;
  for (let i = 0; i < boltCount; i++) {
    const a = (i / boltCount) * Math.PI * 2;
    const by = Math.cos(a) * boltRingR;
    const bz = Math.sin(a) * boltRingR;
    for (const sx of [-1, 1] as const) {
      const bolt = new THREE.CylinderGeometry(boltRadius, boltRadius, 0.006, 6);
      bolt.rotateZ(Math.PI / 2);
      bolt.translate(sx * (hw + 0.002), by, bz);
      parts.push(bolt);
    }
  }

  return mergeGeometriesSafe(parts);
}

/**
 * Builds a telescoping hydraulic/linear actuator strut between two 3D points.
 */
export function createHydraulicStrut(
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  outerRadius: number,
  innerRadius: number,
): { barrel: THREE.BufferGeometry; rod: THREE.BufferGeometry } {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  const totalLen = dir.length();
  const up = new THREE.Vector3(0, 1, 0);
  const quat = new THREE.Quaternion().setFromUnitVectors(up, dir.clone().normalize());

  const barrelLen = totalLen * 0.56;
  const barrel = new THREE.CylinderGeometry(outerRadius, outerRadius * 1.05, barrelLen, 12);
  const collar = new THREE.CylinderGeometry(outerRadius * 1.28, outerRadius * 1.28, 0.012, 12);
  collar.translate(0, barrelLen * 0.5, 0);
  const barrelMerged = mergeGeometriesSafe([barrel, collar]);
  barrelMerged.applyQuaternion(quat);
  const barrelMid = p0.clone().lerp(p1, 0.28);
  barrelMerged.translate(barrelMid.x, barrelMid.y, barrelMid.z);

  const rodLen = totalLen * 0.55;
  const rod = new THREE.CylinderGeometry(innerRadius, innerRadius, rodLen, 10);
  rod.applyQuaternion(quat);
  const rodMid = p0.clone().lerp(p1, 0.72);
  rod.translate(rodMid.x, rodMid.y, rodMid.z);

  return { barrel: barrelMerged, rod };
}

/**
 * Helper: builds a cylinder/capsule aligned along the segment from p0 to p1.
 */
export function createStrutSegment(
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  rBottom: number,
  rTop: number,
  radialSegments = 12,
): THREE.BufferGeometry {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  const len = Math.max(0.002, dir.length());
  const geo = new THREE.CylinderGeometry(rTop, rBottom, len, radialSegments);
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.normalize(),
  );
  geo.applyQuaternion(quat);
  const mid = p0.clone().add(p1).multiplyScalar(0.5);
  geo.translate(mid.x, mid.y, mid.z);
  return geo;
}
