"""Sculpt Emmy's candle drawing: four pillar candles on a blue dish.
Writes frames/blk_XX.npz (blockout, own topology), frames/sc_XXX.npy (sculpt snapshots), final.npz, props.npz, log.json
Units cm. X right, Y depth (camera at -Y), Z up. Table top at z = 0.
"""
import numpy as np, json, time, os
from scipy.spatial import cKDTree
import scipy.sparse as sps
from skimage.measure import marching_cubes

t0 = time.time(); os.makedirs('frames', exist_ok=True)
rng = np.random.default_rng(11)
LOG = []; SNAPS = []; NB = [0]


def log(phase, tool, target, note, **p):
    LOG.append(dict(phase=phase, tool=tool, target=target, note=note,
                    params={k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}))


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1); return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


class Field:
    def __init__(s, h, lo, hi):
        s.h = h; s.o = np.array(lo, np.float32)
        s.xs = [np.arange(lo[i], hi[i], h, dtype=np.float32) for i in range(3)]
        s.F = np.full([len(a) for a in s.xs], 50.0, np.float32)

    def apply(s, f, lo, hi, op, k):
        lo = np.asarray(lo) - 1.5 * k; hi = np.asarray(hi) + 1.5 * k
        sl = []
        for i in range(3):
            a = max(0, int((lo[i] - s.o[i]) / s.h) - 1); b = min(len(s.xs[i]), int((hi[i] - s.o[i]) / s.h) + 2); sl.append(slice(a, b))
        sl = tuple(sl)
        g = [s.xs[0][sl[0]][:, None, None], s.xs[1][sl[1]][None, :, None], s.xs[2][sl[2]][None, None, :]]
        d = f(*g)
        if op == 'union': s.F[sl] = smin(d, s.F[sl], k)
        elif op == 'sub': s.F[sl] = smax(s.F[sl], -d, k)
        elif op == 'int': s.F[sl] = smax(s.F[sl], d, k)


def rcyl(cx, cy, z0, z1, r, rr):
    """rounded vertical cylinder"""
    zc = (z0 + z1) / 2; hh = (z1 - z0) / 2
    def f(x, y, z):
        dx = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r + rr; dz = np.abs(z - zc) - hh + rr
        return np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dz, 0) ** 2) + np.minimum(np.maximum(dx, dz), 0) - rr
    return f, (cx - r - .3, cy - r - .3, z0 - .3), (cx + r + .3, cy + r + .3, z1 + .3)


def torus(cx, cy, z, R, r, sy=1.0):
    def f(x, y, zz):
        q = np.sqrt((x - cx) ** 2 + ((y - cy) / sy) ** 2) - R
        return np.sqrt(q * q + (zz - z) ** 2) - r
    return f, (cx - R - r - .3, cy - R * sy - r - .3, z - r - .3), (cx + R + r + .3, cy + R * sy + r + .3, z + r + .3)


def ellip(c, rr):
    c = np.array(c); rr = np.array(rr)
    def f(x, y, z):
        px, py, pz = x - c[0], y - c[1], z - c[2]
        k0 = np.sqrt((px / rr[0]) ** 2 + (py / rr[1]) ** 2 + (pz / rr[2]) ** 2)
        k1 = np.sqrt((px / rr[0] ** 2) ** 2 + (py / rr[1] ** 2) ** 2 + (pz / rr[2] ** 2) ** 2) + 1e-6
        return k0 * (k0 - 1) / k1
    return f, c - rr - .3, c + rr + .3


# candles read off the drawing (left to right): short-left, tall-slim-back, wide-front, tallest-right
CANDLES = [dict(name='short candle', x=-9.0, y=2.4, r=2.7, h=7.2),
           dict(name='slim candle', x=-4.4, y=5.2, r=3.0, h=15.0),
           dict(name='front candle', x=0.6, y=-1.8, r=4.1, h=11.0),
           dict(name='tall candle', x=7.2, y=4.2, r=4.3, h=19.5)]
DISH_Z = 0.9          # candles stand on the dish floor
PRIMS = []
PRIMS.append(('dish', 'shallow blue dish, slightly oval like the drawing', 'union', 0.3, *rcyl(-0.8, 2.4, 0.0, 0.7, 14.5, 0.35)))
PRIMS.append(('dish rim', 'raised outer rim', 'union', 0.25, *torus(-0.8, 2.4, 0.85, 14.0, 0.55)))
PRIMS.append(('dish inner ring', 'the second ring drawn inside the rim', 'union', 0.2, *torus(-0.8, 2.4, 0.72, 11.6, 0.28)))
for c in CANDLES:
    PRIMS.append((c['name'], f"pillar, {2 * c['r']:.0f} cm wide, {c['h']:.0f} cm tall", 'union', 0.15, *rcyl(c['x'], c['y'], DISH_Z - 0.2, DISH_Z + c['h'], c['r'], 0.5)))
for c in CANDLES:
    top = DISH_Z + c['h']
    PRIMS.append((c['name'] + ' pool', 'scoop the melt pool out of the top', 'sub', 0.35, *ellip((c['x'], c['y'], top + 0.25), (c['r'] * 0.8, c['r'] * 0.8, 1.0))))
LO = (-17.0, -14.0, -0.6); HI = (16.5, 19.5, 21.6)


def build(h, upto=None):
    fld = Field(h, LO, HI)
    for p in PRIMS[:upto]:
        fld.apply(p[4], p[5], p[6], p[2], p[3])
    return fld


def mesh_from(fld):
    v, f, _, _ = marching_cubes(fld.F, 0.0, spacing=(fld.h,) * 3); v = v + fld.o
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    if (fn * (v[f].mean(1) - v.mean(0))).sum(1).mean() < 0: f = f[:, [0, 2, 1]]
    return v.astype(np.float64), f.astype(np.int32)


for i, p in enumerate(PRIMS):
    v, f = mesh_from(build(0.25, i + 1))
    np.savez_compressed(f'frames/blk_{NB[0]:02d}.npz', v=v.astype(np.float32), f=f); NB[0] += 1
    log('blockout', {'union': 'Add', 'sub': 'Carve'}[p[2]], p[0], p[1])
print('blockout', NB[0], round(time.time() - t0, 1))

V, F = mesh_from(build(0.115))
log('mesh', 'Voxel remesh', 'whole set', 'marching cubes at 1.15 mm', verts=len(V), tris=len(F))
print('mesh', len(V), round(time.time() - t0, 1))
E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
A_ = sps.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))).tocsr()
A_ = ((A_ + A_.T) > 0).astype(np.float32); Adj = sps.diags(1 / np.asarray(A_.sum(1)).ravel()) @ A_


def normals(V):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
    for i in range(3): np.add.at(vn, F[:, i], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def fall(t):
    t = np.clip(t, 0, 1); return 1 - (3 * t * t - 2 * t ** 3)


def snap():
    fn = f'frames/sc_{len(SNAPS):03d}.npy'; np.save(fn, V.astype(np.float32)); SNAPS.append([fn, 'A', len(LOG) - 1])


TR = [cKDTree(V)]


def dab(kind, c, R, st, Vn, pinch=0.0, dirv=None):
    i = np.array(TR[0].query_ball_point(c, R), int)
    if len(i) == 0: return
    d = V[i] - c; w = fall(np.linalg.norm(d, axis=1) / R); n = Vn[i]
    if kind == 'draw': V[i] += n * (st * w)[:, None]
    elif kind == 'crease':
        tang = d - (d * n).sum(1)[:, None] * n; V[i] += -n * (st * w)[:, None] - tang * (pinch * w)[:, None] * 0.5
    elif kind == 'grab': V[i] += np.array(dirv) * (st * w)[:, None]
    elif kind == 'flatten':
        V[i, 2] += (c[2] - V[i, 2]) * w * st


def stroke(kind, pts, R, st, spacing=0.3, **kw):
    pts = np.array(pts, float); P = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / (R * spacing))); P += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    Vn = normals(V); TR[0] = cKDTree(V)
    for p in P:
        c = V[TR[0].query(p)[1]].copy()
        dab(kind, c, R, st, Vn, **kw)


def smooth(mask, lam=0.5, it=1):
    for _ in range(it):
        L = Adj @ V - V; V[:] += L * (lam * mask)[:, None]


smooth(np.ones(len(V)), 0.4, 2)
log('sculpt', 'Smooth', 'whole set', 'relax voxel steps', iterations=2); snap()

WICKS = []
for c in CANDLES:
    cx, cy, r, top = c['x'], c['y'], c['r'], DISH_Z + c['h']
    nm = c['name']
    # 1. irregular melted lip: inflate the rim at random spots, push down in others (wax slumps where it melts)
    ang = np.linspace(0, 2 * np.pi, 22, endpoint=False) + rng.uniform(0, 0.3)
    for a in ang:
        p = np.array([cx + np.cos(a) * r * 0.92, cy + np.sin(a) * r * 0.92, top])
        if rng.random() < 0.55:
            stroke('draw', [p], 0.55 + rng.random() * 0.4, 0.06 + rng.random() * 0.08)
        else:
            stroke('grab', [p], 0.9, 0.18 + rng.random() * 0.2, dirv=(0, 0, -1))
    log('detail', 'Draw + Grab', nm + ' lip', 'lumpy melted rim: some spots swell, some slump', dabs=len(ang)); snap()
    # 2. pool: flatten the bowl floor and crease where pool meets the lip
    stroke('flatten', [[cx, cy, top - 0.55]], r * 0.62, 0.6)
    circ = [[cx + np.cos(a) * r * 0.74, cy + np.sin(a) * r * 0.74, top - 0.3] for a in np.linspace(0, 2 * np.pi, 40)]
    stroke('crease', circ, 0.28, 0.06, pinch=0.3)
    log('detail', 'Flatten + Crease', nm + ' pool', 'level the liquid pool, sharpen where it meets the wall', radius_cm=0.28); snap()
    # 3. drips: wax runs down the side from low points of the lip
    nd = 2 + int(rng.integers(0, 3))
    for k in range(nd):
        a = rng.uniform(np.pi * 1.1, np.pi * 1.9)                  # mostly on the side facing us
        L = rng.uniform(0.18, 0.55) * c['h']
        pts = [[cx + np.cos(a) * r * 1.0, cy + np.sin(a) * r * 1.0, top - 0.1 - t * L] for t in np.linspace(0, 1, 12)]
        jit = np.cumsum(rng.normal(0, 0.05, 12))
        pts = [[p[0] + j * np.sin(a), p[1] - j * np.cos(a), p[2]] for p, j in zip(pts, jit)]
        stroke('draw', pts[:9], 0.42, 0.05)
        stroke('draw', pts[8:], 0.55, 0.08)                     # the bead at the end of the run
    log('detail', 'Draw', nm + ' drips', 'wax runs down the side, beading at the end', drips=nd); snap()
    # 4. wick hole
    stroke('crease', [[cx, cy, top - 0.6]], 0.22, 0.25)
    WICKS.append([cx, cy, top - 0.75])
    log('detail', 'Crease', nm + ' wick', 'drill the wick hole', radius_cm=0.22)

# wax surface: faint vertical streaks like the hatching in the drawing
Vn = normals(V)
on_cand = np.zeros(len(V), bool)
for c in CANDLES:
    rr = np.hypot(V[:, 0] - c['x'], V[:, 1] - c['y'])
    on_cand |= (rr > c['r'] - 0.6) & (rr < c['r'] + 0.8) & (V[:, 2] > DISH_Z + 0.3) & (V[:, 2] < DISH_Z + c['h'] - 0.4)
theta = np.arctan2(V[:, 1], V[:, 0])
streak = (np.sin(theta * 160 + np.sin(V[:, 2] * 0.7) * 2.0) * 0.5 + 0.5) ** 6
V += Vn * (0.012 * streak * on_cand)[:, None]
log('detail', 'Draw (texture)', 'candle walls', 'faint vertical wax streaks, echoing the pencil hatching', depth_mm=0.12); snap()
# dish grooves
for R_, nm in ((12.9, 'dish groove'), (10.6, 'inner groove')):
    circ = [[-0.8 + np.cos(a) * R_, 2.4 + np.sin(a) * R_, 0.75] for a in np.linspace(0, 2 * np.pi, 120)]
    stroke('crease', circ, 0.25, 0.035, pinch=0.2)
log('detail', 'Crease', 'dish', 'two grooves between the rims', grooves=2); snap()
smooth(np.ones(len(V)), 0.15, 1)
log('finish', 'Smooth', 'whole set', 'one light polish', strength=0.15); snap()

# labels for materials: dish vs wax
dish = V[:, 2] < DISH_Z + 0.25
rr_any = np.min([np.hypot(V[:, 0] - c['x'], V[:, 1] - c['y']) - c['r'] for c in CANDLES], 0)
dish = dish & (rr_any > -0.2) | (V[:, 2] < 0.45)
np.savez_compressed('final.npz', v=V.astype(np.float32), f=F, dish=dish)
np.savez_compressed('frames/topoA.npz', f=F)
json.dump(dict(log=LOG, snaps=SNAPS, blk_frames=NB[0], wicks=WICKS, candles=CANDLES, dish_z=DISH_Z), open('log.json', 'w'), indent=1)
print('done', len(SNAPS), 'snaps', round(time.time() - t0, 1))
