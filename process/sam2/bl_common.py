import bpy, numpy as np
from mathutils import Vector
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True); return bpy.context.scene
def mk_mat(name,col,rough=0.55,sss=0.0,spec=0.35):
    m=bpy.data.materials.new(name); m.use_nodes=True; p=m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value=(*col,1); p.inputs['Roughness'].default_value=rough
    try: p.inputs['Specular IOR Level'].default_value=spec
    except: pass
    if sss>0:
        p.inputs['Subsurface Weight'].default_value=sss; p.inputs['Subsurface Radius'].default_value=(0.6,0.3,0.2); p.inputs['Subsurface Scale'].default_value=0.4
    return m
CLAY=(0.62,0.47,0.38); HAIR=(0.16,0.13,0.12); SHIRT=(0.06,0.065,0.08); MOUTH=(0.18,0.06,0.06); EYE=(0.80,0.74,0.68); TOOTH=(0.74,0.69,0.6)
def add_mesh(name,V,F,mats=None,matidx=None,smooth=True,coll=None):
    # blender: Z up. our data: y up, z toward viewer -> blender (x, -z, y)
    Vb=np.c_[V[:,0],-V[:,2],V[:,1]]
    me=bpy.data.meshes.new(name); me.from_pydata(Vb.tolist(),[],F.tolist()); me.update()
    me.polygons.foreach_set('use_smooth',[smooth]*len(me.polygons))
    for m in (mats or []): me.materials.append(m)
    if matidx is not None: me.polygons.foreach_set('material_index',matidx.tolist())
    ob=bpy.data.objects.new(name,me); (coll or bpy.context.scene.collection).objects.link(ob); return ob
def look(o,t): o.rotation_euler=(Vector(t)-o.location).to_track_quat('-Z','Y').to_euler()
def studio(sc,target=(0,7,-4),bg=(0.035,0.035,0.04)):
    for nm,loc,e,s in (('key',(-38,-52,38),26000,28),('fill',(48,-40,4),7000,40),('rim',(18,60,34),16000,22),('top',(0,-6,70),5000,30)):
        ld=bpy.data.lights.new(nm,'AREA');ld.energy=e;ld.size=s;l=bpy.data.objects.new(nm,ld);sc.collection.objects.link(l);l.location=loc;look(l,target)
    w=bpy.data.worlds.new('w');sc.world=w;w.use_nodes=True;w.node_tree.nodes['Background'].inputs[0].default_value=(*bg,1)
def camera(sc,loc,target,lens=85):
    cd=bpy.data.cameras.new('cam');cd.lens=lens;cam=bpy.data.objects.new('cam',cd);sc.collection.objects.link(cam);sc.camera=cam;cam.location=loc;look(cam,target);return cam
def cycles(sc,w,h,samples=16):
    sc.render.engine='CYCLES';sc.cycles.samples=samples;sc.cycles.use_denoising=True;sc.render.resolution_x=w;sc.render.resolution_y=h
    sc.cycles.device='CPU'; sc.view_settings.view_transform='AgX'; sc.view_settings.look='AgX - Medium High Contrast'
    sc.render.image_settings.file_format='PNG'
def load_finished(path='finished.npz',V=None):
    d=np.load(path); F=d['f']; V=d['v'] if V is None else V; mat=d['mat']
    mats=[mk_mat('clay',CLAY,0.5,0.15),mk_mat('hair',HAIR,0.8),mk_mat('shirt',SHIRT,0.85),mk_mat('mouth',MOUTH,0.6),mk_mat('pocket',(0.25,0.17,0.14),0.7)]
    fm=np.max(mat[F],1)                      # face material: dominant non-skin label
    from scipy import stats
    fm=stats.mode(mat[F],axis=1).mode.ravel()
    head=add_mesh('head',V,F,mats,fm.astype(int))
    em=mk_mat('eye',EYE,0.35,0.05); tm=mk_mat('teeth',TOOTH,0.35,0.1)
    eR=add_mesh('eyeR',d['eyeR_v'],d['eyeR_f'],[em]); eL=add_mesh('eyeL',d['eyeL_v'],d['eyeL_f'],[em])
    tu=add_mesh('teethU',d['tu_v'],d['tu_f'],[tm]); tl=add_mesh('teethL',d['tl_v'],d['tl_f'],[tm])
    return head,eR,eL,tu,tl
def paint_mat(name='painted',attr='paint',rough=0.55,sss=0.12):
    m=bpy.data.materials.new(name); m.use_nodes=True; nt=m.node_tree; p=nt.nodes['Principled BSDF']
    a=nt.nodes.new('ShaderNodeAttribute'); a.attribute_name=attr; nt.links.new(a.outputs['Color'],p.inputs['Base Color'])
    p.inputs['Roughness'].default_value=rough
    try: p.inputs['Specular IOR Level'].default_value=0.3
    except: pass
    if sss>0: p.inputs['Subsurface Weight'].default_value=sss; p.inputs['Subsurface Radius'].default_value=(0.8,0.35,0.2); p.inputs['Subsurface Scale'].default_value=0.3
    return m
def set_colors(ob,C,attr='paint'):
    me=ob.data; ca=me.color_attributes.new(attr,'FLOAT_COLOR','POINT')
    ca.data.foreach_set('color',np.c_[C,np.ones(len(C))].astype(np.float32).ravel())
def eye_colors(Vv,c):
    d=Vv-c; d/=np.linalg.norm(d,axis=1,keepdims=True); ang=np.degrees(np.arccos(np.clip(d[:,2],-1,1)))
    sclera=np.array([0.36,0.31,0.27]); iris=np.array([0.035,0.022,0.016]); pupil=np.array([0.008,0.006,0.005])
    t=np.clip((ang-24)/3,0,1)[:,None]; col=iris*(1-t)+sclera*t
    col=np.where((ang<9)[:,None],pupil,col); return col
def load_painted(path='finished.npz',paint='paint.npy',V=None):
    d=np.load(path); F=d['f']; V=d['v'] if V is None else V; mat=d['mat']
    from scipy import stats
    fm=stats.mode(mat[F],axis=1).mode.ravel()
    pm=paint_mat(); mm=mk_mat('mouth',MOUTH,0.6)
    head=add_mesh('head',V,F,[pm,mm],(fm==3).astype(int)); set_colors(head,np.load(paint))
    em=paint_mat('eyepaint',rough=0.25,sss=0.0); tm=mk_mat('teeth',TOOTH,0.35,0.1)
    eR=add_mesh('eyeR',d['eyeR_v'],d['eyeR_f'],[em]); set_colors(eR,eye_colors(d['eyeR_v'],d['eyeR_c']))
    eL=add_mesh('eyeL',d['eyeL_v'],d['eyeL_f'],[em]); set_colors(eL,eye_colors(d['eyeL_v'],d['eyeL_c']))
    tu=add_mesh('teethU',d['tu_v'],d['tu_f'],[tm]); tl=add_mesh('teethL',d['tl_v'],d['tl_f'],[tm])
    return head,eR,eL,tu,tl
