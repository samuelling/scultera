"""Render the sculpt time-lapse + expression reel frames in Cycles.
usage: python3 render_frames.py -- <part> [start] [end]     part = tl | reel | test
"""
import bpy, numpy as np, json, sys, math, os
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
part = args[0]
os.makedirs('render', exist_ok=True)
LOGJ = json.load(open('log.json'))
D = np.load('final.npz'); T = np.load('teeth.npz')
TOP = {'A': np.load('frames/topoA.npz')['f'], 'B': D['f']}

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = 8; sc.cycles.use_denoising = True; sc.cycles.max_bounces = 4
sc.cycles.use_adaptive_sampling = True
sc.render.resolution_x = 560; sc.render.resolution_y = 560
sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.36, 0.345, 0.33, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 1.0


def mat(name, col, rough, sss=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*col, 1); p.inputs['Roughness'].default_value = rough
    if sss:
        p.inputs['Subsurface Weight'].default_value = sss; p.inputs['Subsurface Radius'].default_value = (1, 0.5, 0.3); p.inputs['Subsurface Scale'].default_value = 0.15
    return m


CLAY = mat('Clay', (0.50, 0.41, 0.34), 0.55, 0.08)
MOUTH = mat('Mouth', (0.07, 0.025, 0.025), 0.7)
TEETH = mat('Teeth', (0.78, 0.74, 0.66), 0.4)


def look(o, t):
    o.rotation_euler = (Vector(t) - o.location).to_track_quat('-Z', 'Y').to_euler()


for nm, loc, e, s in (('key', (-60, -70, 55), 26000, 40), ('fill', (70, -55, 5), 7000, 60), ('rim', (25, 80, 45), 16000, 30)):
    ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = e; ld.size = s; ld.color = (1, 0.95, 0.88) if nm == 'key' else (0.9, 0.94, 1)
    l = bpy.data.objects.new(nm, ld); sc.collection.objects.link(l); l.location = loc; look(l, (0, 2, -4))
cd = bpy.data.cameras.new('cam'); cd.lens = 70; cam = bpy.data.objects.new('cam', cd); sc.collection.objects.link(cam); sc.camera = cam
def set_cam(az, el, dist, tgt):
    a = math.radians(az); e = math.radians(el)
    cam.location = (tgt[0] + dist * math.sin(a) * math.cos(e), tgt[1] - dist * math.cos(a) * math.cos(e), tgt[2] + dist * math.sin(e))
    look(cam, tgt)


OBJ = {}


def show(name, V, F, mats=None, mi=None):
    if name in OBJ:
        o = OBJ[name]; me = o.data
        if len(me.vertices) == len(V) and len(me.polygons) == len(F):
            me.vertices.foreach_set('co', V.astype(np.float32).ravel()); me.update(); return o
        bpy.data.objects.remove(o); bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(name); me.from_pydata(V.tolist(), [], F.tolist())
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    for m in (mats or [CLAY]):
        me.materials.append(m)
    if mi is not None:
        me.polygons.foreach_set('material_index', mi.astype(np.int32))
    o = bpy.data.objects.new(name, me); sc.collection.objects.link(o); OBJ[name] = o
    return o


def hide(name):
    if name in OBJ:
        OBJ[name].hide_render = True


def unhide(name):
    if name in OBJ:
        OBJ[name].hide_render = False


def render(path):
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)


FB = D['f']; MIN = D['mouth_in']
face_mi = MIN[FB].all(1).astype(np.int32)
VB = D['v'].astype(np.float64)
keys = list(D['keys'])
# head bone: pivot at the top of the neck, weight ramps up the neck
PIV = np.array([0.0, 6.3, -8.5])
def head_w(V):
    return np.clip((V[:, 2] - (-12.0)) / 5.5, 0, 1) ** 1.5


def rotm(pitch, roll, yaw):
    p, r, y = [math.radians(x) for x in (pitch, roll, yaw)]
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def posed(V, w, pose):
    R = rotm(*pose)
    Vr = (V - PIV) @ R.T + PIV
    return V + (Vr - V) * w[:, None]


def expr_state(wts, pose):
    V = VB.copy()
    for k, a in wts.items():
        if a:
            V = V + D['k_' + k] * a
    TV = T['v'].astype(np.float64).copy()
    for k, a in wts.items():
        if a:
            TV[T['lower']] += T['j_' + k] * a
    hw = head_w(VB)
    return posed(V, hw, pose), posed(TV, np.ones(len(TV)), pose)


manifest = []
if part in ('tl', 'test'):
    nblk = LOGJ['blk_frames']; snaps = LOGJ['snaps']
    items = [('blk', i) for i in range(nblk)] + [('sc', i) for i in range(len(snaps))]
    a0 = int(args[1]) if len(args) > 1 else 0; a1 = int(args[2]) if len(args) > 2 else len(items)
    if part == 'test':
        items = [items[-1]]; a0, a1 = 0, 1
    N = len(nblk * [0] + snaps)
    for idx in range(a0, a1):
        kind, i = items[idx]
        t = idx / max(1, N - 1)
        if kind == 'blk':
            B = np.load(f'frames/blk_{i:02d}.npz')
            show('blk', B['v'], B['f']); hide('sculpt'); unhide('blk')
            set_cam(-38 + 10 * t, 12, 95, (0, 3, -6))
            logi = i
        else:
            fn, topo, logi = snaps[i]
            V = np.load(fn)
            F = TOP[topo]
            show('sculpt', V, F, [CLAY]); hide('blk'); unhide('sculpt')
            ph = LOGJ['log'][logi]['phase']
            if ph in ('detail', 'finish'):
                set_cam(-32 + 14 * t, 6, 62, (0, 1, -2))
            else:
                set_cam(-38 + 12 * t, 10, 80, (0, 2, -4))
        out = f'render/tl_{idx:03d}.png'
        render(out)
        manifest.append(dict(frame=idx, kind=kind, log=logi))
        print('rendered', out, flush=True)
    json.dump(manifest, open(f'render/tl_manifest_{a0}_{a1}.json', 'w'))

if part == 'reel':
    poses = LOGJ['poses']; kp = LOGJ['key_photo']
    def P(k, s=0.55):
        if k not in kp:
            return (0, 0, 0)
        p = poses[kp[k]]
        return (-p['pitch'] * s, -p['roll'] * s * 0.6, -p['yaw'] * s)
    seq = [('rest', {}, (0, 0, 0)), ('Smile', {'Smile': 1}, P('Smile')), ('Grin', {'Grin': 1}, P('Grin')),
           ('Crinkle', {'Crinkle': 1}, P('Crinkle')), ('Laugh', {'Laugh': 1}, P('Laugh')),
           ('Brows up', {'Brows up': 1, 'Smile': 0.3}, (6, 0, -8)), ('Shout', {'Shout': 1}, P('Shout')),
           ('Blink', {'Blink': 1, 'Smile': 0.4}, (-4, 3, 10)), ('Beam', {'Beam': 1}, P('Beam')),
           ('Squint', {'Squint': 1}, P('Squint')), ('rest', {}, (0, 0, 0))]
    frames = []
    for (n0, w0, p0), (n1, w1, p1) in zip(seq[:-1], seq[1:]):
        for j in range(7):
            u = j / 7; u = u * u * (3 - 2 * u)
            ks = set(w0) | set(w1)
            frames.append((n1 if u > 0.5 else n0, {k: w0.get(k, 0) * (1 - u) + w1.get(k, 0) * u for k in ks}, tuple(np.array(p0) * (1 - u) + np.array(p1) * u)))
        for j in range(3):
            frames.append((n1, dict(w1), tuple(p1)))
    a0 = int(args[1]) if len(args) > 1 else 0; a1 = int(args[2]) if len(args) > 2 else len(frames)
    for idx in range(a0, min(a1, len(frames))):
        name, wts, pose = frames[idx]
        V, TVp = expr_state(wts, pose)
        show('sculpt', V, FB, [CLAY, MOUTH], face_mi)
        show('teeth', TVp, T['f'], [TEETH])
        set_cam(-24 + 10 * math.sin(idx / len(frames) * math.pi * 2), 5, 58, (0, 1, -2.5))
        render(f'render/reel_{idx:03d}.png')
        manifest.append(dict(frame=idx, name=name, weights={k: round(v, 3) for k, v in wts.items()}, pose=[round(x, 2) for x in pose]))
        print('rendered reel', idx, flush=True)
    json.dump(dict(frames=manifest, total=len(frames)), open(f'render/reel_manifest_{a0}_{a1}.json', 'w'))
