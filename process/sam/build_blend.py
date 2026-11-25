"""Build sam_bust.blend (shape keys, rig, performance action) and export sam_bust.glb for the web."""
import bpy, numpy as np, json, math
from mathutils import Vector, Quaternion, Euler

S = 0.01   # cm -> m
D = np.load('final.npz'); T = np.load('teeth.npz'); LOGJ = json.load(open('log.json'))
keys = list(D['keys'])
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.unit_settings.system = 'METRIC'


def mat(name, col, rough, sss=0.0, bump=False):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; p = nt.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*col, 1); p.inputs['Roughness'].default_value = rough
    if sss:
        p.inputs['Subsurface Weight'].default_value = sss; p.inputs['Subsurface Radius'].default_value = (1, 0.5, 0.3); p.inputs['Subsurface Scale'].default_value = 0.002
    if bump:
        tc = nt.nodes.new('ShaderNodeTexCoord'); nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 900; nz.inputs['Detail'].default_value = 4
        b = nt.nodes.new('ShaderNodeBump'); b.inputs['Strength'].default_value = 0.12; b.inputs['Distance'].default_value = 0.0004
        nt.links.new(tc.outputs['Object'], nz.inputs['Vector']); nt.links.new(nz.outputs['Fac'], b.inputs['Height']); nt.links.new(b.outputs['Normal'], p.inputs['Normal'])
    return m


CLAY = mat('Clay', (0.50, 0.41, 0.34), 0.55, 0.08, bump=True)
MOUTH = mat('Mouth interior', (0.07, 0.025, 0.025), 0.7)
TEETH = mat('Teeth', (0.78, 0.74, 0.66), 0.4)

V = D['v'] * S; F = D['f']
me = bpy.data.meshes.new('Sam'); me.from_pydata(V.tolist(), [], F.tolist())
me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
me.materials.append(CLAY); me.materials.append(MOUTH)
me.polygons.foreach_set('material_index', D['mouth_in'][F].all(1).astype(np.int32))
sam = bpy.data.objects.new('Sam', me); sc.collection.objects.link(sam)
sam.shape_key_add(name='Basis')
for k in keys:
    kb = sam.shape_key_add(name=k)
    kb.data.foreach_set('co', ((D['v'] + D['k_' + k]) * S).astype(np.float32).ravel())

TV = T['v'] * S
tm = bpy.data.meshes.new('Teeth'); tm.from_pydata(TV.tolist(), [], T['f'].tolist())
tm.polygons.foreach_set('use_smooth', [True] * len(tm.polygons)); tm.materials.append(TEETH)
teeth = bpy.data.objects.new('Teeth', tm); sc.collection.objects.link(teeth)
teeth.shape_key_add(name='Basis')
for k in keys:
    kb = teeth.shape_key_add(name=k)
    tv = T['v'].copy(); tv[T['lower']] += T['j_' + k]
    kb.data.foreach_set('co', (tv * S).astype(np.float32).ravel())

# rig: Root (bust) and Head (pivot at the top of the neck)
arm = bpy.data.armatures.new('Rig'); rig = bpy.data.objects.new('Rig', arm); sc.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig; bpy.ops.object.mode_set(mode='EDIT')
r = arm.edit_bones.new('Root'); r.head = (0, 0.05, -0.28); r.tail = (0, 0.055, -0.13)
h = arm.edit_bones.new('Head'); h.head = (0, 0.063, -0.085); h.tail = (0, 0.07, 0.04); h.parent = r
bpy.ops.object.mode_set(mode='OBJECT')
hw = np.clip((D['v'][:, 2] - (-12.0)) / 5.5, 0, 1) ** 1.5
for obj, w in ((sam, hw), (teeth, np.ones(len(T['v'])))):
    gH = obj.vertex_groups.new(name='Head'); gR = obj.vertex_groups.new(name='Root')
    idx = np.arange(len(w))
    for val in np.unique(np.round(w, 3)):
        sel = idx[np.round(w, 3) == val].tolist()
        if val > 0: gH.add(sel, float(val), 'REPLACE')
        if val < 1: gR.add(sel, float(1 - val), 'REPLACE')
    mod = obj.modifiers.new('Armature', 'ARMATURE'); mod.object = rig
    obj.parent = rig

# performance action: expressions + head poses from the photos
poses = LOGJ['poses']; kp = LOGJ['key_photo']
def pose_of(k, s=0.55):
    if k not in kp: return (0, 0, 0)
    p = poses[kp[k]]; return (-p['pitch'] * s, -p['roll'] * s * 0.6, -p['yaw'] * s)
seq = [('Smile', 1), ('Grin', 1), ('Crinkle', 1), ('Laugh', 1), ('Brows up', 1), ('Shout', 1), ('Blink', 1), ('Beam', 1), ('Squint', 1)]
sc.frame_start = 1; f = 1; step = 24
pb = rig.pose.bones['Head']; pb.rotation_mode = 'XYZ'
for kb_owner in (sam, teeth):
    for kb in kb_owner.data.shape_keys.key_blocks[1:]:
        kb.value = 0; kb.keyframe_insert('value', frame=1)
pb.rotation_euler = (0, 0, 0); pb.keyframe_insert('rotation_euler', frame=1)
for name, val in seq:
    f += step
    for kb_owner in (sam, teeth):
        for kb in kb_owner.data.shape_keys.key_blocks[1:]:
            kb.value = val if kb.name == name else 0; kb.keyframe_insert('value', frame=f)
    pi, ro, ya = pose_of(name)
    # bone axes: local Y runs up the neck; map pitch->X, yaw->Y(up), roll->Z(depth)
    pb.rotation_euler = (math.radians(pi), math.radians(ya), math.radians(-ro)); pb.keyframe_insert('rotation_euler', frame=f)
    f += 10
    for kb_owner in (sam, teeth):
        for kb in kb_owner.data.shape_keys.key_blocks[1:]:
            kb.keyframe_insert('value', frame=f)
    pb.keyframe_insert('rotation_euler', frame=f)
f += step
for kb_owner in (sam, teeth):
    for kb in kb_owner.data.shape_keys.key_blocks[1:]:
        kb.value = 0; kb.keyframe_insert('value', frame=f)
pb.rotation_euler = (0, 0, 0); pb.keyframe_insert('rotation_euler', frame=f)
sc.frame_end = f

# camera + lights for the .blend
def look(o, t): o.rotation_euler = (Vector(t) - o.location).to_track_quat('-Z', 'Y').to_euler()
cd = bpy.data.cameras.new('Camera'); cd.lens = 70; cam = bpy.data.objects.new('Camera', cd); sc.collection.objects.link(cam); sc.camera = cam
cam.location = (-0.3, -0.5, 0.05); look(cam, (0, 0.01, -0.025))
for nm, loc, e, s in (('Key', (-0.6, -0.7, 0.55), 2.6, 0.4), ('Fill', (0.7, -0.55, 0.05), 0.7, 0.6), ('Rim', (0.25, 0.8, 0.45), 1.6, 0.3)):
    ld = bpy.data.lights.new(nm, 'AREA'); ld.energy = e * 1000 * 0.0001 * 100; ld.size = s
    l = bpy.data.objects.new(nm, ld); sc.collection.objects.link(l); l.location = loc; look(l, (0, 0.02, -0.04))
w = bpy.data.worlds.new('World'); sc.world = w; w.use_nodes = True; w.node_tree.nodes['Background'].inputs[0].default_value = (0.36, 0.345, 0.33, 1)
sc.render.engine = 'CYCLES'; sc.view_settings.view_transform = 'AgX'
sc.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath='/home/claude/samhead/sam_bust.blend')
print('saved blend')

# web export: no animation (the page drives the face live), morphs + skin
for o in sc.objects: o.select_set(o.name in ('Sam', 'Teeth', 'Rig'))
bpy.ops.export_scene.gltf(filepath='/home/claude/samhead/sam_bust_raw.glb', export_format='GLB', use_selection=True,
                          export_morph=True, export_morph_normal=True, export_skins=True, export_animations=False,
                          export_apply=False, export_cameras=False, export_lights=False)
print('exported glb')
