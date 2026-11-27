import sys, os; sys.path.insert(0,'proc')
import bpy, numpy as np
from bl_common import *
CEN=np.array([0,0,-8.0]); CB=(0.0,8.0,0.0)        # head centre in Blender coords (x, -z, y)
def vcol(ob,name,C):
    ca=ob.data.color_attributes.new(name,'FLOAT_COLOR','POINT'); C=np.c_[C,np.ones(len(C))] if C.shape[1]==3 else C; ca.data.foreach_set('color',C.astype(np.float32).ravel())
def mat4(mode):
    m=bpy.data.materials.new(mode); m.use_nodes=True; nt=m.node_tree; p=nt.nodes['Principled BSDF']; p.inputs['Roughness'].default_value=0.6
    try: p.inputs['Specular IOR Level'].default_value=0.25
    except: pass
    if mode=='clay':
        a=nt.nodes.new('ShaderNodeAttribute'); a.attribute_name='clay'; nt.links.new(a.outputs['Color'],p.inputs['Base Color']); return m
    a=nt.nodes.new('ShaderNodeAttribute'); a.attribute_name='paint'; r=nt.nodes.new('ShaderNodeAttribute'); r.attribute_name='rest'
    sep=nt.nodes.new('ShaderNodeSeparateXYZ'); cmb=nt.nodes.new('ShaderNodeCombineXYZ'); env=nt.nodes.new('ShaderNodeTexEnvironment')
    env.image=bpy.data.images.load(os.path.abspath('texture.jpg')); mix=nt.nodes.new('ShaderNodeMix'); mix.data_type='RGBA'
    L=nt.links.new; L(r.outputs['Vector'],sep.inputs[0]); L(sep.outputs['Y'],cmb.inputs['X']); L(sep.outputs['X'],cmb.inputs['Y']); L(sep.outputs['Z'],cmb.inputs['Z']); L(cmb.outputs[0],env.inputs['Vector'])
    L(a.outputs['Alpha'],mix.inputs[0]); L(a.outputs['Color'],mix.inputs[6]); L(env.outputs['Color'],mix.inputs[7]); L(mix.outputs[2],p.inputs['Base Color']); return m
def load4(mode='paint',fade=1.0):
    d=np.load('finished4.npz'); V=d['v']; F=d['f']; P=np.load('work/paint_v.npy').copy(); CL=np.load('work/clay_v.npy'); m=mat4(mode); tm=mk_mat('teeth',(0.6,0.55,0.47),0.35,0.1)
    if fade<1: P[:,:3]=CL*(1-fade)+P[:,:3]*fade; P[:,3]*=fade
    head=add_mesh('head',V,F,[m]); vcol(head,'paint',P); vcol(head,'clay',CL)
    ra=head.data.attributes.new('rest','FLOAT_VECTOR','POINT'); ra.data.foreach_set('vector',((np.c_[V[:,0],-V[:,2],V[:,1]]-np.array(CB))*[-1,-1,1]).astype(np.float32).ravel())
    objs=[head]
    for nm in ('eyeR','eyeL'):
        Ve=d[nm+'_v']; o=add_mesh(nm,Ve,d[nm+'_f'],[m]); vcol(o,'paint',np.c_[d[nm+'_col'] if fade>=1 else np.tile([0.62,0.55,0.48],(len(Ve),1))*(1-fade)+d[nm+'_col']*fade,np.zeros(len(Ve))]); vcol(o,'clay',np.tile([0.62,0.55,0.48],(len(Ve),1)))
        o.data.attributes.new('rest','FLOAT_VECTOR','POINT'); objs.append(o)
    objs.append(add_mesh('teethU',d['tu_v'],d['tu_f'],[tm])); objs.append(add_mesh('teethL',d['tl_v'],d['tl_f'],[tm])); return objs
if __name__=='__main__':
    from PIL import Image
    out=sys.argv[1]; rows=[]
    for mode in ('clay','paint'):
        sc=reset(); load4(mode); studio(sc); cycles(sc,400,480,12); ims=[]
        for nm,(loc,tgt) in {'front':((0,-105,-2),(0,6,-3)),'q':((-58,-86,3),(0,6,-3)),'side':((-105,6,-2),(0,6,-3)),'back':((40,110,2),(0,6,-3))}.items():
            for o in list(bpy.data.objects):
                if o.type=='CAMERA': bpy.data.objects.remove(o)
            camera(sc,loc,tgt,lens=85); sc.render.filepath='/tmp/p4_%s.png'%nm; bpy.ops.render.render(write_still=True); ims.append(Image.open('/tmp/p4_%s.png'%nm).convert('RGB'))
        r=Image.new('RGB',(1600,480)); [r.paste(im,(400*i,0)) for i,im in enumerate(ims)]; rows.append(r)
    c=Image.new('RGB',(1600,960)); c.paste(rows[0],(0,0)); c.paste(rows[1],(0,480)); c.resize((1200,720)).save(out)
