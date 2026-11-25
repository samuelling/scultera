import bpy
bpy.ops.wm.open_mainfile(filepath='/home/claude/samhead/sam_bust.blend')
sc=bpy.context.scene; sc.frame_set(1)
for o in sc.objects: o.select_set(o.name in ('Sam','Teeth'))
bpy.ops.export_scene.gltf(filepath='/home/claude/samhead/sam_bust_raw.glb', export_format='GLB', use_selection=True,
    export_morph=True, export_morph_normal=True, export_skins=False, export_animations=False, export_apply=False, export_cameras=False, export_lights=False)
print('exported')
