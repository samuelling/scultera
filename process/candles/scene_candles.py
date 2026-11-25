"""Candles: build the Blender scene, render time-lapse frames + finale, export .blend and .glb.
usage: python3 scene_candles.py -- tl | finale | blend
"""
import bpy, numpy as np, json, sys, math, os
from mathutils import Vector

part = sys.argv[sys.argv.index('--') + 1]
os.makedirs('render', exist_ok=True)
L = json.load(open('log.json')); D = np.load('final.npz'); F0 = D['f']
rng = np.random.default_rng(5)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.use_denoising = True; sc.cycles.max_bounces = 6
sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
sc.render.resolution_x = 560; sc.render.resolution_y = 700


def look(o, t):
    o.rotation_euler = (Vector(t) - o.location).to_track_quat('-Z', 'Y').to_euler()


def principled(name, col, rough, sss=0.0, sss_col=None):
    m = bpy.data.materials.new(name); m.use_nodes = True; p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*col, 1); p.inputs['Roughness'].default_value = rough
    if sss:
        p.inputs['Subsurface Weight'].default_value = sss; p.inputs['Subsurface Radius'].default_value = sss_col or (1.0, 0.6, 0.35)
        p.inputs['Subsurface Scale'].default_value = 0.9
    return m, p


CLAY, _ = principled('Clay', (0.50, 0.41, 0.34), 0.55)
WAX, _ = principled('Wax', (0.93, 0.89, 0.80), 0.42, sss=0.55)
GLAZE, gp = principled('Blue glaze', (0.20, 0.27, 0.55), 0.18)
gp.inputs['Coat Weight'].default_value = 0.6
WICK, _ = principled('Wick', (0.05, 0.035, 0.025), 0.8)
# flame: emission with a hot core
FLAME = bpy.data.materials.new('Flame'); FLAME.use_nodes = True; nt = FLAME.node_tree
for n in list(nt.nodes): nt.nodes.remove(n)
out = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission'); lw = nt.nodes.new('ShaderNodeLayerWeight')
ramp = nt.nodes.new('ShaderNodeValToRGB'); mix = nt.nodes.new('ShaderNodeMixShader'); tr = nt.nodes.new('ShaderNodeBsdfTransparent')
lw.inputs['Blend'].default_value = 0.45
ramp.color_ramp.elements[0].color = (1.0, 0.93, 0.7, 1); ramp.color_ramp.elements[1].color = (1.0, 0.45, 0.08, 1); ramp.color_ramp.elements[1].position = 0.85
nt.links.new(lw.outputs['Facing'], ramp.inputs[0]); nt.links.new(ramp.outputs[0], em.inputs['Color']); em.inputs['Strength'].default_value = 22
nt.links.new(lw.outputs['Facing'], mix.inputs[0]); nt.links.new(em.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
nt.links.new(mix.outputs[0], out.inputs['Surface'])
SMOKE = bpy.data.materials.new('Smoke'); SMOKE.use_nodes = True; nt = SMOKE.node_tree
for n in list(nt.nodes): nt.nodes.remove(n)
o2 = nt.nodes.new('ShaderNodeOutputMaterial'); m2 = nt.nodes.new('ShaderNodeMixShader'); t2 = nt.nodes.new('ShaderNodeBsdfTransparent'); d2 = nt.nodes.new('ShaderNodeBsdfDiffuse')
d2.inputs['Color'].default_value = (0.3, 0.3, 0.32, 1); m2.inputs[0].default_value = 0.45
nt.links.new(t2.outputs[0], m2.inputs[1]); nt.links.new(d2.outputs[0], m2.inputs[2]); nt.links.new(m2.outputs[0], o2.inputs['Surface'])
# table (pinkish wood) and canvas wall
WOOD = bpy.data.materials.new('Table wood'); WOOD.use_nodes = True; nt = WOOD.node_tree; p = nt.nodes['Principled BSDF']
wv = nt.nodes.new('ShaderNodeTexWave'); wv.inputs['Scale'].default_value = 0.18; wv.inputs['Distortion'].default_value = 6; wv.inputs['Detail'].default_value = 3
wv.bands_direction = 'X'
rp = nt.nodes.new('ShaderNodeValToRGB'); rp.color_ramp.elements[0].color = (0.33, 0.17, 0.14, 1); rp.color_ramp.elements[1].color = (0.52, 0.32, 0.27, 1)
nt.links.new(wv.outputs['Fac'], rp.inputs[0]); nt.links.new(rp.outputs[0], p.inputs['Base Color']); p.inputs['Roughness'].default_value = 0.6
WALL = bpy.data.materials.new('Canvas wall'); WALL.use_nodes = True; nt = WALL.node_tree; p = nt.nodes['Principled BSDF']
p.inputs['Base Color'].default_value = (0.80, 0.78, 0.74, 1); p.inputs['Roughness'].default_value = 0.95
ck = nt.nodes.new('ShaderNodeTexChecker'); ck.inputs['Scale'].default_value = 900; bm = nt.nodes.new('ShaderNodeBump'); bm.inputs['Strength'].default_value = 0.25
nt.links.new(ck.outputs['Fac'], bm.inputs['Height']); nt.links.new(bm.outputs['Normal'], p.inputs['Normal'])


def add_mesh(name, V, F, mats, mi=None, smooth=True):
    me = bpy.data.meshes.new(name); me.from_pydata(np.asarray(V).tolist(), [], np.asarray(F).tolist())
    if smooth: me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    for m in mats: me.materials.append(m)
    if mi is not None: me.polygons.foreach_set('material_index', mi.astype(np.int32))
    o = bpy.data.objects.new(name, me); sc.collection.objects.link(o); return o


# table + wall
add_mesh('Table', [(-60, -40, 0), (60, -40, 0), (60, 30, 0), (-60, 30, 0)], [(0, 1, 2, 3)], [WOOD], smooth=False)
add_mesh('Wall', [(-60, 30, 0), (60, 30, 0), (60, 30, 70), (-60, 30, 70)], [(0, 1, 2, 3)], [WALL], smooth=False)
cd = bpy.data.cameras.new('Camera'); cd.lens = 52; cam = bpy.data.objects.new('Camera', cd); sc.collection.objects.link(cam); sc.camera = cam
cam.location = (-3, -64, 24); look(cam, (-0.5, 3, 9))
w = bpy.data.worlds.new('World'); sc.world = w; w.use_nodes = True; w.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.48, 0.45, 1)
w.node_tree.nodes['Background'].inputs[1].default_value = 0.35
kd = bpy.data.lights.new('Window', 'AREA'); kd.energy = 32000; kd.size = 45; kd.color = (0.9, 0.93, 1.0)
kl = bpy.data.objects.new('Window', kd); sc.collection.objects.link(kl); kl.location = (-45, -45, 55); look(kl, (0, 3, 6))


# props: wicks, flames, smoke ribbons (modelled, not sculpted)
def tube(path, r0, r1, seg=8):
    path = np.asarray(path); n = len(path); Vt = []; Ft = []
    for i in range(n):
        t = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]; t /= np.linalg.norm(t) + 1e-9
        a = np.cross(t, [1, 0, 0]); a = a if np.linalg.norm(a) > 0.1 else np.cross(t, [0, 1, 0]); a /= np.linalg.norm(a); b = np.cross(t, a)
        r = r0 + (r1 - r0) * i / max(1, n - 1)
        for k in range(seg):
            th = 2 * np.pi * k / seg; Vt.append(path[i] + r * (np.cos(th) * a + np.sin(th) * b))
    for i in range(n - 1):
        for k in range(seg):
            a0 = i * seg + k; a1 = i * seg + (k + 1) % seg; Ft += [(a0, a0 + seg, a1), (a1, a0 + seg, a1 + seg)]
    return np.array(Vt), np.array(Ft)


def flame_mesh(base, h, rw, seg=24, rings=18):
    Vt = []; Ft = []
    for i in range(rings + 1):
        u = i / rings; r = rw * (math.sin(math.pi * min(1, u * 1.25)) ** 0.7) * (1 - u) ** 0.35 if u < 1 else 0
        for k in range(seg):
            th = 2 * np.pi * k / seg; Vt.append((base[0] + r * math.cos(th), base[1] + r * math.sin(th), base[2] + u * h))
    for i in range(rings):
        for k in range(seg):
            a0 = i * seg + k; a1 = i * seg + (k + 1) % seg; Ft += [(a0, a0 + seg, a1), (a1, a0 + seg, a1 + seg)]
    return np.array(Vt), np.array(Ft)


PROPS = []
for i, (c, wk) in enumerate(zip(L['candles'], L['wicks'])):
    wk = np.array(wk); lean = rng.uniform(-0.25, 0.25, 2)
    wpath = [wk + np.array([lean[0] * t * t, lean[1] * t * t, t * 1.15]) for t in np.linspace(0, 1, 8)]
    wv, wf = tube(wpath, 0.09, 0.07, 6); PROPS.append(('Wick ' + c['name'], wv, wf, WICK))
    tip = wpath[-1]
    fh = 2.4 + 0.25 * i % 2; fv, ff = flame_mesh(tip - np.array([0, 0, 0.55]), fh, 0.55); PROPS.append(('Flame ' + c['name'], fv, ff, FLAME))
    # smoke: two twisting strands rising from above the flame, like the pencil ribbons
    for s in range(2):
        ph = rng.uniform(0, 6.3); amp = rng.uniform(0.5, 1.1); top = rng.uniform(5.5, 8.5)
        sp = [tip + np.array([amp * (t / top) * math.sin(t * 1.3 + ph + s * 1.7), 0.3 * math.cos(t + ph), fh + 0.2 + t]) for t in np.linspace(0, top, 40)]
        sv, sf = tube(sp, 0.035, 0.11, 6); PROPS.append((f'Smoke {c["name"]} {s}', sv, sf, SMOKE))
    ld = bpy.data.lights.new('Glow ' + c['name'], 'POINT'); ld.energy = 120; ld.color = (1.0, 0.62, 0.25); ld.shadow_soft_size = 0.6
    lo = bpy.data.objects.new('Glow ' + c['name'], ld); sc.collection.objects.link(lo); lo.location = tuple(tip + np.array([0, 0, 0.9]))
np.savez_compressed('props.npz', **{f'v{i}': p[1] for i, p in enumerate(PROPS)}, **{f'f{i}': p[2] for i, p in enumerate(PROPS)},
                    names=np.array([p[0] for p in PROPS]), mats=np.array([p[3].name for p in PROPS]))
PROP_OBJS = [add_mesh(n, v, f, [m]) for n, v, f, m in PROPS]
GLOWS = [o for o in sc.objects if o.name.startswith('Glow')]


def set_props(visible):
    for o in PROP_OBJS + GLOWS: o.hide_render = not visible


wax_mi = D['dish'][F0].all(1).astype(np.int32)

if part == 'tl':
    sc.cycles.samples = 6; sc.render.resolution_x = 480; sc.render.resolution_y = 600
    set_props(False); obj = None; man = []
    nb = L['blk_frames']; snaps = L['snaps']; items = [('blk', i) for i in range(nb)] + [('sc', i) for i in range(len(snaps))]
    for idx, (kind, i) in enumerate(items):
        if kind == 'blk':
            B = np.load(f'frames/blk_{i:02d}.npz'); V, F = B['v'], B['f']
        else:
            V = np.load(snaps[i][0]); F = F0
        if obj is not None:
            me = obj.data; bpy.data.objects.remove(obj); bpy.data.meshes.remove(me)
        obj = add_mesh('sculpt', V, F, [CLAY])
        t = idx / (len(items) - 1)
        cam.location = (-14 + 12 * t, -62, 26 - 3 * t); look(cam, (-0.5, 3, 9))
        sc.render.filepath = f'render/tl_{idx:03d}.png'; bpy.ops.render.render(write_still=True); print('rendered', idx, flush=True)

if part in ('finale', 'blend'):
    sculpt = add_mesh('Candles', D['v'], F0, [WAX, GLAZE], wax_mi)
    set_props(True)
    if part == 'finale':
        sc.cycles.samples = 64; sc.render.resolution_x = 900; sc.render.resolution_y = 1125
        sc.render.filepath = 'render/finale.png'; bpy.ops.render.render(write_still=True); print('rendered finale')
        # "lighting the candles" frames for the end of the time-lapse: material reveal, then flames
        sc.cycles.samples = 8; sc.render.resolution_x = 480; sc.render.resolution_y = 600
        for j, (wax, props) in enumerate(((False, False), (True, False), (True, True))):
            sculpt.data.materials[0] = WAX if wax else CLAY; sculpt.data.materials[1] = GLAZE if wax else CLAY
            set_props(props)
            sc.render.filepath = f'render/lit_{j}.png'; bpy.ops.render.render(write_still=True); print('rendered lit', j)
    else:
        bpy.ops.wm.save_as_mainfile(filepath='/home/claude/candles/candles.blend')
        for o in sc.objects: o.select_set(o.type == 'MESH' and o.name not in ('Table', 'Wall'))
        bpy.ops.export_scene.gltf(filepath='/home/claude/candles/candles_raw.glb', export_format='GLB', use_selection=True,
                                  export_cameras=False, export_lights=False, export_apply=False)
        print('saved blend + glb')
