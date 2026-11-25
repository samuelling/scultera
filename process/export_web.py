import bpy, numpy as np, os
OUT='/home/claude/gecko/web/models/'
def gl(path,**extra):
    kw=dict(filepath=path,export_format='GLB',export_draco_mesh_compression_enable=True,export_draco_mesh_compression_level=7,
            export_draco_position_quantization=14,export_draco_normal_quantization=10,export_draco_color_quantization=10,
            export_cameras=False,export_lights=False,export_apply=True)
    kw.update(extra)
    bpy.ops.export_scene.gltf(**kw)
# clay stages
for nm in ('base','stage_grab','stage_crease'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    D=np.load(f'/home/claude/gecko/{nm}.npz')
    me=bpy.data.meshes.new(nm);me.from_pydata(D['v'].tolist(),[],D['f'].tolist());me.polygons.foreach_set('use_smooth',[True]*len(me.polygons))
    ob=bpy.data.objects.new('Gecko',me);bpy.context.scene.collection.objects.link(ob)
    gl(OUT+nm.replace('stage_','')+'.glb',export_materials='NONE')
# final
bpy.ops.wm.open_mainfile(filepath='/home/claude/gecko/sad_gecko.blend')
sc=bpy.context.scene
sc.cycles.samples=16;sc.cycles.device='CPU'
eye=bpy.data.objects['Eye.R'];em=eye.data.materials[0];nt=em.node_tree
img=bpy.data.images.new('eye_iris',1024,512);img.colorspace_settings.name='sRGB'
tn=nt.nodes.new('ShaderNodeTexImage');tn.image=img;nt.nodes.active=tn
for o in sc.objects: o.select_set(False)
eye.select_set(True);bpy.context.view_layer.objects.active=eye
sc.render.bake.use_pass_direct=False;sc.render.bake.use_pass_indirect=False;sc.render.bake.use_pass_color=True;sc.render.bake.margin=4
bpy.ops.object.bake(type='DIFFUSE',pass_filter={'COLOR'})
img.filepath_raw=OUT+'eye_iris.png';img.file_format='PNG';img.save()
P=nt.nodes['Principled BSDF']
for l in list(P.inputs['Base Color'].links): nt.links.remove(l)
nt.links.new(tn.outputs['Color'],P.inputs['Base Color'])
# skin -> simple vertex-colour material
sm=bpy.data.objects['Gecko'].data.materials[0];nt=sm.node_tree
for n in list(nt.nodes):
    if n.type not in ('BSDF_PRINCIPLED','OUTPUT_MATERIAL'): nt.nodes.remove(n)
P=nt.nodes['Principled BSDF'];ca=nt.nodes.new('ShaderNodeVertexColor');ca.layer_name='Col';nt.links.new(ca.outputs['Color'],P.inputs['Base Color'])
P.inputs['Roughness'].default_value=0.55;P.inputs['Subsurface Weight'].default_value=0;P.inputs['Sheen Weight'].default_value=0
for nm in ('Floor','Key','Fill','Rim','Bounce'):
    if nm in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[nm])
for o in list(sc.objects):
    if o.type=='CAMERA': bpy.data.objects.remove(o)
tr=bpy.data.objects['TearTrail']
for o in sc.objects: o.select_set(False)
tr.select_set(True);bpy.context.view_layer.objects.active=tr;bpy.ops.object.convert(target='MESH')
try: gl(OUT+'final.glb',export_vertex_color='ACTIVE')
except TypeError as e:
    print('retry',e);gl(OUT+'final.glb')
print('OK')
