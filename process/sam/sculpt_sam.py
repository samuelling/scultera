"""Sculpt a bust of Sam from multi-photo face landmarks.
Writes: frames/blk_XX.npz (blockout frames, own topology), frames/sc_XXX.npy (sculpt snapshots, shared topology),
base_topology.npz, final.npz (verts, faces, shape keys, labels), log.json
Units: centimetres. Axes: X = subject's left(+)/right(-) as seen in image, Y = depth (camera at -Y), Z = up.
"""
import numpy as np, json, time, os
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import scipy.sparse as sps
from skimage.measure import marching_cubes

t0 = time.time()
os.makedirs('frames', exist_ok=True)
AL = np.load('aligned.npz', allow_pickle=True)
ID = AL['ident']                      # identity landmarks, cm
NAMES = list(AL['names'])
rng = np.random.default_rng(7)
LOG = []
FRAME = {'blk': 0, 'sc': 0}


def log(phase, tool, target, note, **params):
    LOG.append(dict(phase=phase, tool=tool, target=target, note=note,
                    params={k: (round(v, 3) if isinstance(v, float) else v) for k, v in params.items()},
                    t=round(time.time() - t0, 1)))


# ------------------------------------------------------------------ SDF blockout
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


class Field:
    def __init__(s, h, box):
        s.h = h
        s.o = np.array(box[0], np.float32)
        s.xs = [np.arange(box[0][i], box[1][i], h, dtype=np.float32) for i in range(3)]
        s.F = np.full([len(a) for a in s.xs], 50.0, np.float32)

    def sub(s, lo, hi):
        sl = []
        for i in range(3):
            a = max(0, int((lo[i] - s.o[i]) / s.h) - 1)
            b = min(len(s.xs[i]), int((hi[i] - s.o[i]) / s.h) + 2)
            sl.append(slice(a, b))
        g = [s.xs[0][sl[0]][:, None, None], s.xs[1][sl[1]][None, :, None], s.xs[2][sl[2]][None, None, :]]
        return tuple(sl), g

    def apply(s, d_fn, lo, hi, op, k):
        sl, g = s.sub(np.asarray(lo) - 1.5 * k, np.asarray(hi) + 1.5 * k)
        if sl[0].stop <= sl[0].start or sl[1].stop <= sl[1].start or sl[2].stop <= sl[2].start:
            return
        d = d_fn(*g)
        if op == 'union':
            s.F[sl] = smin(d, s.F[sl], k)
        elif op == 'sub':
            s.F[sl] = smax(s.F[sl], -d, k)
        elif op == 'int':
            s.F[sl] = smax(s.F[sl], d, k)


def ellip_fn(c, r, Rm=None):
    c = np.array(c, np.float32); r = np.array(r, np.float32)
    def f(x, y, z):
        px, py, pz = x - c[0], y - c[1], z - c[2]
        if Rm is not None:
            px, py, pz = (Rm[0, 0] * px + Rm[1, 0] * py + Rm[2, 0] * pz,
                          Rm[0, 1] * px + Rm[1, 1] * py + Rm[2, 1] * pz,
                          Rm[0, 2] * px + Rm[1, 2] * py + Rm[2, 2] * pz)
        k0 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2)
        k1 = np.sqrt((px / r[0] ** 2) ** 2 + (py / r[1] ** 2) ** 2 + (pz / r[2] ** 2) ** 2) + 1e-6
        return k0 * (k0 - 1) / k1
    m = float(r.max()) + 0.3
    return f, c - m, c + m


def capsule_fn(a, b, ra, rb):
    a = np.array(a, np.float32); b = np.array(b, np.float32); ab = b - a; L2 = float(ab @ ab)
    def f(x, y, z):
        px, py, pz = x - a[0], y - a[1], z - a[2]
        t = np.clip((px * ab[0] + py * ab[1] + pz * ab[2]) / L2, 0, 1)
        dx, dy, dz = px - t * ab[0], py - t * ab[1], pz - t * ab[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * t)
    m = max(ra, rb) + 0.3
    return f, np.minimum(a, b) - m, np.maximum(a, b) + m


def rotx(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def roty(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def rotz(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


# landmark-derived anchor points
L = ID
mouth_c = (L[13] + L[14]) / 2
INNER_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
lipline = (L[INNER_UP] + L[INNER_LO]) / 2
lip_x = lipline[:, 0]; order = np.argsort(lip_x)
lip_z_of_x = interp1d(lip_x[order], lipline[order, 2], bounds_error=False, fill_value=(lipline[order, 2][0], lipline[order, 2][-1]))
lip_y_of_x = interp1d(lip_x[order], lipline[order, 1], bounds_error=False, fill_value=(lipline[order, 1][0], lipline[order, 1][-1]))
mouth_half = (lip_x.max() - lip_x.min()) / 2

PRIMS = []  # (name, kind, fn, lo, hi, op, k, note)


def P(name, note, op, k, fnlohi):
    f, lo, hi = fnlohi
    PRIMS.append((name, note, op, k, f, lo, hi))


P('cranium', 'one big egg for the skull', 'union', 1.0, ellip_fn((0, 8.5, 2.4), (7.9, 9.6, 8.8)))
P('face mass', 'the front plane of the face', 'union', 1.0, ellip_fn((0, 2.9, -3.6), (6.7, 4.7, 6.8)))
for sg in (-1, 1):
    P(f'cheekbone {"R" if sg < 0 else "L"}', 'zygomatic arch, tucked under the face', 'union', 1.2, ellip_fn((sg * 4.7, 3.3, -1.9), (2.2, 2.4, 1.7)))
for sg in (-1, 1):
    ja = np.array([sg * 4.9, 4.6, -7.4])
    P(f'jaw line {"R" if sg < 0 else "L"}', 'mandible, from under the ear to the chin', 'union', 1.0, capsule_fn(ja, L[152] + np.array([sg * 1.6, 1.4, 0.9]), 1.45, 1.15))
chin = L[152] + np.array([0, 1.0, 1.1])
P('chin', 'mental protuberance', 'union', 0.7, ellip_fn(chin, (2.2, 1.8, 1.8)))
P('muzzle', 'the dental arch the lips wrap around', 'union', 0.6, ellip_fn((0, mouth_c[1] + 1.4, mouth_c[2] + 0.2), (3.3, 2.3, 2.5)))
P('nose', 'bridge to tip wedge', 'union', 0.35, capsule_fn((0, L[6][1] + 0.35, L[6][2]), (L[1][0], L[1][1] + 0.55, L[1][2] + 0.1), 0.55, 0.95))
for sg, i in ((-1, 64), (1, 294)):
    P(f'nostril wing {"R" if sg < 0 else "L"}', 'alar lobe', 'union', 0.3, ellip_fn(L[i] + np.array([-sg * 0.2, 0.35, 0.1]), (0.75, 0.7, 0.6)))
P('brow ridge', 'supraorbital bar', 'union', 0.7, capsule_fn((-3.3, L[105][1] + 1.2, L[105][2] - 0.1), (3.3, L[334][1] + 1.2, L[334][2] - 0.1), 0.9, 0.9))
for sg in (-1, 1):
    ear_c = np.array([sg * 7.35, 8.6, -1.2])
    Rm = rotz(sg * -28) @ rotx(-12)                 # flare out ~28 deg (his ears stand out)
    P(f'ear {"R" if sg < 0 else "L"}', 'flattened disc, flared ~28 deg', 'union', 0.25, ellip_fn(ear_c, (0.55, 1.9, 3.0), Rm))
    bowl_c = ear_c + Rm @ np.array([sg * 0.45, -0.25, -0.35], np.float32)
    P(f'ear bowl {"R" if sg < 0 else "L"}', 'concha carved into the ear disc', 'sub', 0.12, ellip_fn(bowl_c, (0.42, 1.25, 1.9), Rm))
P('neck', 'cylinder, slightly forward-leaning', 'union', 1.6, capsule_fn((0, 6.6, -5.5), (0, 5.4, -19.5), 6.2, 6.6))
P('trapezius', 'shoulder slope', 'union', 2.0, ellip_fn((0, 7.0, -21.0), (13.5, 6.0, 4.2)))
P('chest', 'upper chest block', 'union', 1.8, ellip_fn((0, 3.8, -24.0), (12.0, 6.2, 6.2)))
P('bust cut', 'flat bottom of the bust', 'int', 0.4, (lambda x, y, z: -27.8 - z, np.array([-20, -10, -40.]), np.array([20, 25, -20.])))
P('bust sides', 'trim the shoulders', 'int', 0.8, (lambda x, y, z: np.abs(x) - 12.8 + 0 * y + 0 * z, np.array([-20, -10, -40.]), np.array([20, 25, 0.])))
# the measured face: a slab whose front surface is the 9-photo landmark heightfield (target only)
from scipy.interpolate import griddata
from matplotlib.path import Path as MPath
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
_gx = np.arange(-9.5, 9.5, 0.1); _gz = np.arange(-14.0, 9.0, 0.1)
_GX, _GZ = np.meshgrid(_gx, _gz, indexing='ij')
_pts = np.c_[_GX.ravel(), _GZ.ravel()]
_yc = griddata(L[:468][:, [0, 2]], L[:468][:, 1], _pts, method='cubic')
_yn = griddata(L[:468][:, [0, 2]], L[:468][:, 1], _pts, method='nearest')
YF = np.where(np.isfinite(_yc), _yc, _yn).reshape(_GX.shape)
_op = MPath(L[OVAL][:, [0, 2]])
_ins = _op.contains_points(_pts).reshape(_GX.shape)
from scipy.ndimage import distance_transform_edt, gaussian_filter
_din = distance_transform_edt(_ins) * 0.1           # cm inside the oval
YF = gaussian_filter(YF, 0.6)
_gy = np.hypot(*np.gradient(YF, 0.1))
WOV = np.clip(_din / 1.4, 0, 1) * np.clip((2.6 - _gy) / 1.4, 0, 1)
WOV = gaussian_filter(WOV, 8)
from scipy.interpolate import RegularGridInterpolator as RGI
_yf = RGI((_gx, _gz), YF, bounds_error=False, fill_value=None)
_wo = RGI((_gx, _gz), WOV, bounds_error=False, fill_value=0.0)
def face_fn(x, y, z):
    X, Y, Z = np.broadcast_arrays(x, y, z)
    q = np.c_[X.ravel(), Z.ravel()]
    yf = _yf(q).reshape(X.shape); w = _wo(q).reshape(X.shape)
    d = np.maximum(yf - Y, Y - (yf + 7.0))
    return (d + (1 - w) * 8.0).astype(np.float32)
def cut_fn(x, y, z):
    X, Y, Z = np.broadcast_arrays(x, y, z)
    q = np.c_[X.ravel(), Z.ravel()]
    yf = _yf(q).reshape(X.shape); w = _wo(q).reshape(X.shape)
    return ((yf - Y) - (1 - w) * 8.0).astype(np.float32)
P('face trim', 'shave the blockout back to your face surface', 'int', 0.25, (cut_fn, np.array([-9.4, -6.5, -13.9]), np.array([9.4, 10.0, 8.9])))
P('face (9 photos)', 'your averaged face surface', 'union', 0.9, (face_fn, np.array([-9.4, -6.5, -13.9]), np.array([9.4, 10.0, 8.9])))
FACE_PRIMS = {len(PRIMS) - 2, len(PRIMS) - 1}
FACE_PRIM = len(PRIMS) - 1
# mouth: slit along the measured lip line + oral cavity, so expressions can open the jaw
slit_pts = []
for xv in np.linspace(-mouth_half * 0.97, mouth_half * 0.97, 90):
    zf = float(lip_z_of_x(xv)); yf = float(lip_y_of_x(xv))
    slit_pts.append(((xv, yf - 1.2, zf), (xv * 0.8, yf + 1.4, zf)))
def slit_fn(x, y, z):
    d = None
    for a, b in slit_pts:
        f, _, _ = capsule_fn(a, b, 0.045, 0.06)
        v = f(x, y, z); d = v if d is None else np.minimum(d, v)
    return d
P('mouth slit', 'cut along your lip line so the jaw can open', 'sub', 0.03,
  (slit_fn, np.array([-mouth_half - 0.5, mouth_c[1] - 1.6, mouth_c[2] - 0.6]), np.array([mouth_half + 0.5, mouth_c[1] + 1.8, mouth_c[2] + 0.6])))
cav_c = np.array([0, mouth_c[1] + 2.3, mouth_c[2] - 0.1])
P('oral cavity', 'hollow behind the lips', 'sub', 0.25, ellip_fn(cav_c, (2.3, 1.6, 1.25)))
teeth = []
for row, dz in (('upper', 0.5), ('lower', -0.5)):
    for i, u in enumerate(np.linspace(-1, 1, 12)):
        ang = u * 1.15
        p = cav_c + np.array([1.6 * np.sin(ang), -1.05 * np.cos(ang) + 0.2, dz])
        w = 0.24 if abs(u) < 0.3 else 0.2
        teeth.append((row, p, w))
TEETH = teeth   # built as separate meshes below

BOX = (np.array([-15.0, -6.5, -29.0]), np.array([15.0, 18.0, 13.6]))


def build(h, upto=None, face=False):
    fld = Field(h, BOX)
    for i, (name, note, op, k, f, lo, hi) in enumerate(PRIMS[:upto]):
        if i in FACE_PRIMS and not face:
            continue
        fld.apply(f, lo, hi, op, k)
    return fld


def mesh_from(fld):
    v, f, _, _ = marching_cubes(fld.F, 0.0, spacing=(fld.h,) * 3)
    v = v + fld.o
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    if (fn * (v[f].mean(1) - v.mean(0))).sum(1).mean() < 0:
        f = f[:, [0, 2, 1]]
    return v.astype(np.float64), f.astype(np.int32)


# blockout frames (coarse, one per primitive group)
group_bounds = []
cur = None
for i, (name, note, op, k, f, lo, hi) in enumerate(PRIMS):
    g = name.split(' (')[0].rstrip(' RL').strip()
    if g != cur:
        group_bounds.append([g, i, i + 1, note, op]); cur = g
    else:
        group_bounds[-1][2] = i + 1
for g, a, b, note, op in group_bounds:
    if g.startswith('face (9') or g.startswith('face trim'):
        continue
    if b < 3:   # need at least cranium+face to mesh meaningfully; still record
        pass
    fld = build(0.3, b)
    v, f = mesh_from(fld)
    np.savez_compressed(f'frames/blk_{FRAME["blk"]:02d}.npz', v=v.astype(np.float32), f=f)
    FRAME['blk'] += 1
    log('blockout', {'union': 'Add', 'sub': 'Carve', 'int': 'Trim'}[op], g, note, primitives=b - a)
print('blockout frames', FRAME['blk'], round(time.time() - t0, 1))

# fine mesh
fld = build(0.12)
V, F = mesh_from(fld)
del fld
TF = build(0.12, face=True)
np.savez_compressed('frames/topoA.npz', f=F)
print('fine mesh', len(V), len(F), round(time.time() - t0, 1))
log('mesh', 'Voxel remesh', 'whole bust', 'marching cubes at 1.2 mm', verts=len(V), tris=len(F))

# ------------------------------------------------------------------ brush engine
from scipy.sparse.csgraph import connected_components
TOPO = {'name': 'A'}
def setup_topology(name):
    global E, Adj, is_tooth, tooth_upper, tooth_lower
    TOPO['name'] = name
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A_ = sps.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))).tocsr()
    A_ = ((A_ + A_.T) > 0).astype(np.float32)
    Adj = sps.diags(1 / np.asarray(A_.sum(1)).ravel()) @ A_
    ncomp, comp = connected_components(Adj, directed=False)
    main = np.argmax(np.bincount(comp))
    is_tooth = comp != main
    tooth_upper = is_tooth & (V[:, 2] > cav_c[2])
    tooth_lower = is_tooth & ~tooth_upper
    print('topology', name, len(V), 'verts; components', ncomp, 'teeth verts', int(is_tooth.sum()))


def normals(V):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for i in range(3):
        np.add.at(vn, F[:, i], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def fall(t):
    t = np.clip(t, 0, 1); return 1 - (3 * t * t - 2 * t ** 3)


setup_topology('A')


def side_of(P):
    """+1 = upper (above lip line), -1 = lower; only meaningful near the mouth"""
    zl = lip_z_of_x(np.clip(P[:, 0], lip_x.min(), lip_x.max()))
    return np.where(P[:, 2] < zl, -1, 1)


def near_mouth(P):
    d = np.sqrt(((P[:, 0]) / (mouth_half + 1.6)) ** 2 + ((P[:, 2] - mouth_c[2]) / 2.2) ** 2)
    return (d < 1) & (P[:, 1] < mouth_c[1] + 3.5)


SNAPS = []
def snap():
    fn = f'frames/sc_{FRAME["sc"]:03d}.npy'
    np.save(fn, V.astype(np.float32)); SNAPS.append([fn, TOPO['name'], len(LOG) - 1])
    FRAME['sc'] += 1


def frontal_hits(targets, Vn, search=1.5):
    """for each landmark target, the nearest front-facing surface vertex (ICP-style correspondence)"""
    front = np.where((Vn[:, 1] < 0.15) & ~is_tooth & (V[:, 2] > -14))[0]
    tr = cKDTree(V[front])
    d, j = tr.query(targets, k=1)
    return front[j], d


def field_from(src_pts, deltas, sigma, src_side=None, kmax=12, cutoff=3.0, mask=None):
    """Smooth displacement field: Gaussian-weighted (Shepard) interpolation of landmark deltas, fading
    to zero away from the landmark cloud. Lips: vertices only listen to landmarks on their own side."""
    tr = cKDTree(src_pts)
    idx = np.arange(len(V)) if mask is None else np.where(mask)[0]
    d, j = tr.query(V[idx], k=kmax, distance_upper_bound=cutoff * sigma * 2.5)
    valid = np.isfinite(d)
    j = np.where(valid, j, 0)
    w = np.where(valid, np.exp(-(d / sigma) ** 2), 0.0)
    if src_side is not None:
        vs = side_of(V[idx]); nm = near_mouth(V[idx])
        conflict = (src_side[j] != vs[:, None]) & nm[:, None] & (src_side[j] != 0)
        w = np.where(conflict, 0, w)
    ws = w.sum(1)
    disp = (w[:, :, None] * deltas[j]).sum(1) / np.maximum(ws, 1e-9)[:, None]
    reach = np.clip(ws / (ws + 0.15), 0, 1)                  # fade where few landmarks are near
    out = np.zeros_like(V); out[idx] = disp * reach[:, None]
    out[is_tooth] = 0
    return out


LM_SIDE = np.where(near_mouth(ID), side_of(ID), 0)
# lips: force landmark side labels by name
LIP_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 191, 80, 81, 82, 13, 312, 311, 310, 415, 78, 308]
LIP_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 95, 88, 178, 87, 14, 317, 402, 318, 324]
LM_SIDE[LIP_UPPER] = 1; LM_SIDE[LIP_LOWER] = -1

REGIONS = {
    'forehead': [10, 151, 9, 108, 337, 69, 299, 67, 297, 109, 338, 104, 333, 103, 332, 54, 284, 21, 251],
    'brows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 64, 294, 48, 278, 129, 358, 102, 331, 49, 279, 220, 440, 45, 275, 44, 274, 240, 460],
    'cheeks': [50, 280, 116, 345, 117, 346, 118, 347, 101, 330, 36, 266, 205, 425, 123, 352, 147, 376, 187, 411, 207, 427, 216, 436],
    'mouth': LIP_UPPER + LIP_LOWER + [57, 287, 186, 410, 92, 322, 165, 391, 167, 393, 164, 18, 200, 83, 313],
    'chin & jaw': [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397, 288, 361, 175, 199, 171, 396, 140, 369],
    'face outline': [234, 127, 162, 21, 454, 356, 389, 251, 93, 323, 137, 366, 177, 401],
}

# ------------------------------------------------------------------ sculpt: relax
for _ in range(2):
    m = ~is_tooth
    L_ = Adj @ V - V
    V[m] += 0.5 * L_[m]
log('sculpt', 'Smooth', 'whole bust', 'relax the voxel stair-steps', iterations=2)
snap()

# ------------------------------------------------------------------ sculpt: Grab passes onto the measured face
from scipy.interpolate import RegularGridInterpolator as RGI3
TFi = RGI3(tuple(TF.xs), TF.F, bounds_error=False, fill_value=50.0)
def project(P, it=6):
    P = P.copy(); e = 0.06
    for _ in range(it):
        d = TFi(P)
        g = np.stack([(TFi(P + np.eye(3)[k] * e) - TFi(P - np.eye(3)[k] * e)) / (2 * e) for k in range(3)], 1)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
        P -= np.clip(d, -3, 3)[:, None] * g
    return P
T_ = project(V)
T_[is_tooth] = V[is_tooth]
DISP = T_ - V
reg_pts = {r: ID[np.array(ids)] for r, ids in REGIONS.items()}
def region_masks(P):
    W = np.stack([np.exp(-(cKDTree(reg_pts[r]).query(P)[0] / 1.4) ** 2) for r in REGIONS], 1)
    rest = np.clip(1 - W.sum(1), 0, 1)
    return np.c_[W, rest] / np.maximum(W.sum(1, keepdims=True) + rest[:, None], 1e-6)
RM = region_masks(V)
names_r = list(REGIONS) + ['head & jaw blend']
done = np.zeros(len(V))
for pass_i, frac in enumerate((0.45, 0.8, 1.0)):
    for r_i, reg in enumerate(names_r):
        target_amt = frac * RM[:, r_i]
        step = target_amt - done * RM[:, r_i] / np.maximum(RM.sum(1), 1e-6) * 0  # per-region bookkeeping below
        prev = getattr(region_masks, 'acc', None)
        if prev is None:
            region_masks.acc = np.zeros((len(V), len(names_r)))
        inc = np.clip(frac * RM[:, r_i] - region_masks.acc[:, r_i], 0, None)
        region_masks.acc[:, r_i] += inc
        V += DISP * inc[:, None]
        if pass_i == 2 or reg in ('nose', 'mouth', 'eyes', 'chin & jaw', 'head & jaw blend'):
            log('fit', 'Grab', reg, f'pull onto your averaged 9-photo face (pass {pass_i + 1} of 3)',
                max_move_cm=float(np.linalg.norm(DISP * inc[:, None], axis=1).max()))
            snap()
print('fit done', round(time.time() - t0, 1))
# stretched triangles after a big Grab: voxel remesh onto the target, as you would in Blender
V, F = mesh_from(TF)
del TF, TFi
np.savez_compressed('frames/topoB.npz', f=F)
setup_topology('B')
log('fit', 'Voxel remesh', 'whole bust', 'rebuild even topology after the big Grab moves', verts=len(V), tris=len(F))
snap()

Vn = normals(V)
hit, dist = frontal_hits(ID[:468], Vn)
resid = np.linalg.norm(ID[:468] - V[hit], axis=1)
print('fit residual mean %.3f cm, 95%% %.3f cm' % (resid.mean(), np.percentile(resid, 95)))
log('fit', 'Check', 'all 468 landmarks', 'distance from sculpt surface to your averaged landmarks',
    mean_mm=float(resid.mean() * 10), p95_mm=float(np.percentile(resid, 95) * 10))

# ------------------------------------------------------------------ detail brushes
def surf_pt(p):
    j = cKDTree(V).query(p)[1]; return V[j].copy()


def stroke(kind, pts, R, st, pinch=0.0, spacing=0.3, falloff=None, proj=True):
    pts = np.array(pts, float)
    Pp = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / (R * spacing)))
        Pp += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    Vn = normals(V); fr = np.where((Vn[:, 1] < 0.6) & ~is_tooth)[0]; tr = cKDTree(V[fr][:, [0, 2]]) if proj else cKDTree(V); tr3 = cKDTree(V)
    for p in Pp:
        if proj:
            cand = fr[tr.query_ball_point(p[[0, 2]], 0.08)]
            if len(cand) == 0: continue
            c = V[cand[np.argmin(V[cand, 1])]].copy()
        else:
            c = V[tr.query(p)[1]].copy()
        i = np.array(tr3.query_ball_point(c, R), int)
        i = i[~is_tooth[i]]
        if len(i) == 0:
            continue
        d = V[i] - c; w = fall(np.linalg.norm(d, axis=1) / R); n = Vn[i]
        if kind == 'draw':
            V[i] += n * (st * w)[:, None]
        elif kind == 'crease':
            tang = d - (d * n).sum(1)[:, None] * n
            V[i] += -n * (st * w)[:, None] - tang * (pinch * w)[:, None] * 0.5
        elif kind == 'inflate':
            V[i] += n * (st * w)[:, None]


def lm_path(ids, push=(0, 0, 0)):
    return [ID[i] + np.array(push) for i in ids]


# eyelids
UPPER_R = [33, 246, 161, 160, 159, 158, 157, 173, 133]
UPPER_L = [263, 466, 388, 387, 386, 385, 384, 398, 362]
LOWER_R = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LOWER_L = [263, 249, 390, 373, 374, 380, 381, 382, 362]
for nm, ids in (('upper lid R', UPPER_R), ('upper lid L', UPPER_L)):
    stroke('crease', lm_path(ids), 0.14, 0.04, pinch=0.25)
    stroke('draw', lm_path(ids, (0, -0.02, 0.17)), 0.16, 0.025)
    log('detail', 'Crease + Draw', nm, 'lid margin, then roll the lid edge forward', radius_cm=0.14)
    snap()
for nm, ids in (('lower lid R', LOWER_R), ('lower lid L', LOWER_L)):
    stroke('crease', lm_path(ids), 0.12, 0.03, pinch=0.3)
    stroke('inflate', lm_path(ids, (0, 0, -0.35)), 0.3, 0.025)
    log('detail', 'Crease + Inflate', nm, 'lower lid line and the soft smile-bag under it', radius_cm=0.12)
    snap()
# carved irises and drilled pupils (classic bust treatment)
for nm, ctr, ring in (('iris R', 468, [469, 470, 471, 472]), ('iris L', 473, [474, 475, 476, 477])):
    c = ID[ctr]; r = np.mean([np.linalg.norm(ID[k] - c) for k in ring]) * 0.95
    circle = [c + np.array([r * np.cos(a), 0, r * np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 28)]
    stroke('crease', circle, 0.07, 0.03, pinch=0.2)
    stroke('crease', [c + np.array([0, 0, 0.03])], 0.12, 0.09, pinch=0.0)
    log('detail', 'Crease', nm, 'incise the iris ring, drill the pupil', ring_radius_cm=float(r))
    snap()
# crow's feet (his smile crinkles)
for nm, corner, sg in (('crow\'s feet R', 33, -1), ('crow\'s feet L', 263, 1)):
    c = ID[corner]
    for ang in (-35, -10, 15, 38):
        a = np.radians(ang)
        d = np.array([sg * np.cos(a), 0.3, np.sin(a)])
        stroke('crease', [c + d * 0.35, c + d * 1.35], 0.06, 0.018, pinch=0.3)
    log('detail', 'Crease', nm, 'four fanned lines from the outer eye corner', lines=4)
    snap()
# nose
for nm, ids in (('nostril R', [98, 64, 48, 115]), ('nostril L', [327, 294, 278, 344])):
    stroke('crease', lm_path(ids, (0, 0.1, -0.05)), 0.18, 0.035, pinch=0.15)
    log('detail', 'Crease', nm, 'tuck under the alar wing', radius_cm=0.12)
snap()
for nm, ids in (('alar groove R', [129, 203, 98]), ('alar groove L', [358, 423, 327])):
    stroke('crease', lm_path(ids), 0.12, 0.035, pinch=0.4)
    log('detail', 'Crease', nm, 'groove where the nose wing meets the cheek', radius_cm=0.12)
snap()
# lips
OUT_UP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
OUT_LO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
stroke('crease', lm_path(OUT_UP), 0.09, 0.022, pinch=0.3)
stroke('crease', lm_path(OUT_LO), 0.1, 0.018, pinch=0.25)
log('detail', 'Crease', 'vermilion border', 'the edge of the lips, upper and lower', radius_cm=0.09)
snap()
stroke('inflate', lm_path([37, 0, 267]), 0.35, 0.03); stroke('inflate', lm_path([84, 17, 314]), 0.4, 0.04)
stroke('crease', lm_path([0, 164]), 0.1, 0.03, pinch=0.2)
log('detail', 'Inflate + Crease', 'lips and philtrum', 'fill out the lips, groove the philtrum', radius_cm=0.35)
snap()
for nm, ids, sg in (('nasolabial fold R', [129, 203, 206, 216, 57], -1), ('nasolabial fold L', [358, 423, 426, 436, 287], 1)):
    path = [ID[i] + np.array([sg * 0.15, 0, 0]) for i in ids]
    stroke('crease', path, 0.22, 0.025, pinch=0.15)
    stroke('draw', [p + np.array([sg * 0.25, -0.05, 0.05]) for p in path], 0.35, 0.015)
    log('detail', 'Crease + Draw', nm, 'smile line from nose wing to mouth corner, with the cheek rolling over it', radius_cm=0.18)
    snap()
for nm, i in (('cheek apple R', 50), ('cheek apple L', 280)):
    stroke('inflate', [ID[i]], 1.0, 0.12)
    log('detail', 'Inflate', nm, 'round the cheek where it lifts when you smile', radius_cm=1.0)
snap()
stroke('crease', [L[200] + np.array([-0.6, 0, -0.1]), L[200] + np.array([0.6, 0, -0.1])], 0.2, 0.03, pinch=0.2)
log('detail', 'Crease', 'chin crease', 'mentolabial sulcus under the lower lip', radius_cm=0.2)
snap()
# forehead lines (faint)
for k, dz in enumerate((1.2, 2.0, 2.8)):
    pts = [np.array([x, 0, L[9][2] + dz]) for x in np.linspace(-3.2, 3.2, 9)]
    for p in pts:
        p[1] = surf_pt(p + np.array([0, -5, 0]))[1] if False else p[1]
    stroke('crease', [np.array([x, -3.0, L[9][2] + dz]) for x in np.linspace(-3.4, 3.4, 9)], 0.1, 0.006, pinch=0.15)
log('detail', 'Crease', 'forehead lines', 'three faint horizontal lines', lines=3)
snap()
# neck and collarbones
for sg in (-1, 1):
    stroke('draw', [np.array([sg * 4.2, 2.6, -8.5]), np.array([sg * 1.4, -0.4, -18.5])], 0.9, 0.06, proj=False)
log('detail', 'Draw', 'neck tendons', 'sternocleidomastoids, ear to collarbone notch', radius_cm=0.9)
snap()
for sg in (-1, 1):
    stroke('draw', [np.array([sg * 1.6, -2.0, -20.8]), np.array([sg * 7.5, -0.5, -20.2]), np.array([sg * 11.0, 2.0, -19.8])], 0.55, 0.12, proj=False)
stroke('crease', [np.array([0, -2.0, -19.2])], 0.6, 0.12, proj=False)
log('detail', 'Draw + Crease', 'collarbones', 'clavicles and the notch between them', radius_cm=0.55)
snap()

# hair: raised mass above the hairline, then strand grooves
Vn = normals(V)
hair_line_z = L[10][2] + 1.9
dist_ear = np.minimum(np.linalg.norm(V[:, [0, 2]] - np.array([-7.3, -1.2]), axis=1), np.linalg.norm(V[:, [0, 2]] - np.array([7.3, -1.2]), axis=1))
temple_recess = 0.9 * np.exp(-((np.abs(V[:, 0]) - 4.0) / 1.4) ** 2)
hz = hair_line_z + temple_recess
front_mask = np.clip((V[:, 2] - hz) / 1.5, 0, 1)
side_mask = np.clip((V[:, 1] - 4.8) / 0.6, 0, 1) * np.clip((V[:, 2] + 1.0) / 0.6, 0, 1) * np.clip((np.abs(V[:, 0]) - 5.0) / 0.5, 0, 1)
back_mask = np.clip((V[:, 1] - 9.0) / 0.8, 0, 1) * np.clip((V[:, 2] + 4.5) / 0.8, 0, 1)
hair = np.clip(np.maximum(front_mask * (V[:, 1] < 12), np.maximum(side_mask, back_mask)), 0, 1)
hair *= (dist_ear > 2.8)
hair[is_tooth] = 0
hair = np.clip(hair + 0 * hair, 0, 1)
for k in range(3):
    hair = 0.5 * hair + 0.5 * (Adj @ hair)
V += Vn * (0.1 * hair)[:, None]
log('detail', 'Draw (masked)', 'hair mass', 'buzz-cut layer, hairline receding at the temples', thickness_cm=0.1)
snap()
crown = np.array([0, 9.5, 11.5])
hv = np.where(hair > 0.5)[0]
seeds = V[rng.choice(hv, 5500, replace=False)]
segs = []
for s in seeds:
    d = s - crown; d[2] = 0; d = d / (np.linalg.norm(d) + 1e-9); d[2] = -0.35
    segs.append([s + d * t for t in np.linspace(0, 0.32, 5)])
SP = np.array(segs).reshape(-1, 3)
dd, _ = cKDTree(SP).query(V, distance_upper_bound=0.06)
groove = np.where(np.isfinite(dd), fall(dd / 0.05), 0) * hair ** 2
Vn = normals(V)
V -= Vn * (0.03 * groove)[:, None]
log('detail', 'Crease (stipple)', 'hair texture', 'thousands of short grooves flowing from the crown', grooves=len(seeds))
snap()

# final polish
m = (~is_tooth)
L_ = Adj @ V - V
V[m] += 0.15 * L_[m]
log('finish', 'Smooth', 'whole bust', 'one light polish pass', strength=0.15)
snap()
print('sculpt done', FRAME['sc'], 'snapshots', round(time.time() - t0, 1))

# ------------------------------------------------------------------ expression shape keys
Vn = normals(V)
hit_all, _ = frontal_hits(ID[:468], Vn)
src = V[hit_all]
KEYS = {}
def add_key(name, dl, note, sigma=0.75):
    dl = np.vstack([dl[:468]])
    disp = field_from(src, dl, sigma, LM_SIDE[:468])
    disp *= np.clip((V[:, 2] - (L[152][2] - 2.2)) / 1.6, 0, 1)[:, None] * np.clip((7.5 - V[:, 1]) / 2.0, 0, 1)[:, None]
    # teeth: upper stay with skull, lower follow the chin/lower-lip landmarks
    jaw = dl[[17, 14, 152, 18, 200, 175]].mean(0)
    disp[tooth_lower] = jaw * 0.85
    KEYS[name] = (disp.astype(np.float32), note)
    log('expression', 'Shape key', name, note, max_move_cm=float(np.linalg.norm(disp, axis=1).max()))

src_photo = {'Smile': ('p01_front_smile', 'closed-mouth smile (graduation selfie)'),
             'Grin': ('p04_teeth_smile_hair', 'big toothy smile (Blue Hills)'),
             'Crinkle': ('p03_squint_grin_tilt', 'squinting grin (Back Bay)'),
             'Beam': ('p05_smile_3q_tilt', 'wide smile (on the boat)'),
             'Laugh': ('p07_laugh_open_tilt', 'open-mouth laugh (on the boat)'),
             'Shout': ('p09_shout_surprise', 'wide-eyed shout (O\'Connell Park)'),
             'Squint': ('p06_squint_3q_bw', 'sunny squint (black-and-white film)')}
for k, (ph, note) in src_photo.items():
    add_key(k, AL['d_' + ph], note)
# brows-only surprise from the shout (upper face) and a synthetic blink
d = AL['d_p09_shout_surprise'].copy()
upper_w = np.clip((ID[:, 2] - (L[1][2] + 0.5)) / 1.5, 0, 1)[:, None]
add_key('Brows up', d * upper_w, 'eyebrows lifted, from the shout photo\'s upper face only')
bl = np.zeros((478, 3))
for up, lo in zip([246, 161, 160, 159, 158, 157, 173, 466, 388, 387, 386, 385, 384, 398],
                  [7, 163, 144, 145, 153, 154, 155, 249, 390, 373, 374, 380, 381, 382]):
    bl[up] = (ID[lo] - ID[up]) * 0.92 + np.array([0, -0.05, 0])
add_key('Blink', bl, 'upper lids down onto the lower lids (built, not photographed)', sigma=0.35)

# teeth: two separate arches; lower arch follows the jaw in each expression
tv, tf = [], []
for row, p, w in TEETH:
    th = np.linspace(0, np.pi, 9); ph = np.linspace(0, 2 * np.pi, 13)[:-1]
    base = len(tv) if False else sum(len(x) for x in tv)
    vv = np.array([[w * np.sin(a) * np.cos(b), 0.2 * np.sin(a) * np.sin(b), 0.28 * np.cos(a)] for a in th for b in ph]) + p
    ff = []
    for i in range(len(th) - 1):
        for j in range(len(ph)):
            a0 = i * len(ph) + j; a1 = i * len(ph) + (j + 1) % len(ph); b0 = a0 + len(ph); b1 = a1 + len(ph)
            ff += [[a0, b0, a1], [a1, b0, b1]]
    tv.append((row, vv, np.array(ff)))
TV = []; TF_ = []; TROW = []; off = 0
for row, vv, ff in tv:
    TV.append(vv); TF_.append(ff + off); TROW += [row == 'lower'] * len(vv); off += len(vv)
TV = np.vstack(TV); TF_ = np.vstack(TF_); TROW = np.array(TROW)
JAW = {}
for k in KEYS:
    disp = KEYS[k][0]
    # jaw motion = mean displacement of the chin/lower-lip skin
    chin_idx = cKDTree(V).query(ID[[17, 14, 152, 18, 200, 175]])[1]
    JAW[k] = (disp[chin_idx].mean(0) * 0.9).astype(np.float32)
np.savez_compressed('teeth.npz', v=TV.astype(np.float32), f=TF_, lower=TROW, keys=np.array(list(KEYS)), **{'j_' + k: JAW[k] for k in KEYS})
rel = (V - cav_c) / np.array([2.9, 2.3, 1.9])
mouth_in = (np.linalg.norm(rel, axis=1) < 1.0) & (V[:, 1] > lip_y_of_x(np.clip(V[:, 0], lip_x.min(), lip_x.max())) + 0.25)
for _ in range(2):
    mouth_in = mouth_in | ((Adj @ mouth_in.astype(np.float32)) > 0.5) & (np.linalg.norm(rel, axis=1) < 1.1)
np.savez_compressed('final.npz', v=V.astype(np.float32), f=F, tooth_upper=tooth_upper, tooth_lower=tooth_lower, mouth_in=mouth_in,
                    keys=np.array(list(KEYS)), **{'k_' + k: KEYS[k][0] for k in KEYS})
json.dump(dict(snaps=SNAPS, log=LOG, blk_frames=FRAME['blk'], sc_frames=FRAME['sc'], key_notes={k: KEYS[k][1] for k in KEYS},
               poses=json.load(open('poses.json')), key_photo={k: v[0] for k, v in src_photo.items()}),
          open('log.json', 'w'), indent=1)
print('done', round(time.time() - t0, 1), 's; keys:', list(KEYS))
