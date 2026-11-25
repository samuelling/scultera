import bpy, numpy as np, math, sys
from mathutils import Vector, Matrix
bpy.ops.wm.read_factory_settings(use_empty=True)
sc=bpy.context.scene
D=np.load('sculpt.npz');V=D['v'];Fc=D['f'];col=D['col'];E=D['eyes']
me=bpy.data.meshes.new('Gecko');me.from_pydata(V.tolist(),[],Fc.tolist())
me.polygons.foreach_set('use_smooth',[True]*len(me.polygons))
ca=me.color_attributes.new('Col','FLOAT_COLOR','POINT')
ca.data.foreach_set('color',np.hstack([col,np.ones((len(col),1),np.float32)]).ravel())
g=bpy.data.objects.new('Gecko',me);sc.collection.objects.link(g)
def N(nt,t,loc=(0,0)):
    n=nt.nodes.new(t);n.location=loc;return n
# skin
m=bpy.data.materials.new('Gecko Skin');m.use_nodes=True;nt=m.node_tree;L=nt.links
P=nt.nodes['Principled BSDF']
cattr=N(nt,'ShaderNodeVertexColor');cattr.layer_name='Col'
tc=N(nt,'ShaderNodeTexCoord')
nz=N(nt,'ShaderNodeTexNoise');nz.inputs['Scale'].default_value=9;nz.inputs['Detail'].default_value=6
L.new(tc.outputs['Object'],nz.inputs['Vector'])
mr=N(nt,'ShaderNodeMapRange');mr.inputs[3].default_value=0.85;mr.inputs[4].default_value=1.12;L.new(nz.outputs['Fac'],mr.inputs[0])
mul=N(nt,'ShaderNodeMix');mul.data_type='RGBA';mul.blend_type='MULTIPLY';mul.inputs['Factor'].default_value=1
L.new(cattr.outputs['Color'],mul.inputs[6]);L.new(mr.outputs[0],mul.inputs[7])
L.new(mul.outputs[2],P.inputs['Base Color'])
rr=N(nt,'ShaderNodeMapRange');rr.inputs[3].default_value=0.42;rr.inputs[4].default_value=0.62;L.new(nz.outputs['Fac'],rr.inputs[0]);L.new(rr.outputs[0],P.inputs['Roughness'])
P.inputs['Subsurface Weight'].default_value=0.15;P.inputs['Subsurface Radius'].default_value=(1.0,0.55,0.3);P.inputs['Subsurface Scale'].default_value=0.08
P.inputs['Sheen Weight'].default_value=0.15
vor=N(nt,'ShaderNodeTexVoronoi');vor.feature='DISTANCE_TO_EDGE';vor.inputs['Scale'].default_value=55;L.new(tc.outputs['Object'],vor.inputs['Vector'])
bump=N(nt,'ShaderNodeBump');bump.inputs['Strength'].default_value=0.35;bump.inputs['Distance'].default_value=0.012
L.new(vor.outputs['Distance'],bump.inputs['Height']);L.new(bump.outputs['Normal'],P.inputs['Normal'])
me.materials.append(m)
# eye material
em=bpy.data.materials.new('Gecko Eye');em.use_nodes=True;nt=em.node_tree;L=nt.links;P=nt.nodes['Principled BSDF']
tc=N(nt,'ShaderNodeTexCoord');sx=N(nt,'ShaderNodeSeparateXYZ');L.new(tc.outputs['Object'],sx.inputs[0])
def math_(op,a,b=None,v=None):
    n=N(nt,'ShaderNodeMath');n.operation=op
    if isinstance(a,float): n.inputs[0].default_value=a
    else: L.new(a,n.inputs[0])
    if b is not None:
        if isinstance(b,float): n.inputs[1].default_value=b
        else: L.new(b,n.inputs[1])
    return n.outputs[0]
def mrange(x,f0,f1,t0,t1):
    n=N(nt,'ShaderNodeMapRange');n.clamp=True;L.new(x,n.inputs[0]);n.inputs[1].default_value=f0;n.inputs[2].default_value=f1;n.inputs[3].default_value=t0;n.inputs[4].default_value=t1;return n.outputs[0]
ax=math_('ABSOLUTE',sx.outputs['X']);ay=math_('ABSOLUTE',math_('ADD',sx.outputs['Y'],0.1))
pup=math_('MULTIPLY',mrange(ax,0.10,0.14,1,0),mrange(ay,0.55,0.72,1,0))
cx=N(nt,'ShaderNodeCombineXYZ');L.new(sx.outputs['X'],cx.inputs[0]);L.new(sx.outputs['Y'],cx.inputs[1])
ln=N(nt,'ShaderNodeVectorMath');ln.operation='LENGTH';L.new(cx.outputs[0],ln.inputs[0])
limb=mrange(ln.outputs['Value'],0.62,0.95,0,1)
inz=N(nt,'ShaderNodeTexNoise');inz.inputs['Scale'].default_value=5;inz.inputs['Detail'].default_value=8;L.new(tc.outputs['Object'],inz.inputs['Vector'])
cr=N(nt,'ShaderNodeValToRGB');L.new(inz.outputs['Fac'],cr.inputs[0])
cr.color_ramp.elements[0].position=0.35;cr.color_ramp.elements[0].color=(0.16,0.15,0.13,1)
cr.color_ramp.elements[1].position=0.65;cr.color_ramp.elements[1].color=(0.42,0.40,0.35,1)
vv=N(nt,'ShaderNodeTexVoronoi');vv.feature='DISTANCE_TO_EDGE';vv.inputs['Scale'].default_value=11;L.new(tc.outputs['Object'],vv.inputs['Vector'])
vein=mrange(vv.outputs['Distance'],0.0,0.05,1,0)
mv=N(nt,'ShaderNodeMix');mv.data_type='RGBA';L.new(vein,mv.inputs['Factor']);L.new(cr.outputs[0],mv.inputs[6]);mv.inputs[7].default_value=(0.02,0.018,0.015,1)
mx=N(nt,'ShaderNodeMix');mx.data_type='RGBA';L.new(limb,mx.inputs['Factor']);L.new(mv.outputs[2],mx.inputs[6]);mx.inputs[7].default_value=(0.04,0.035,0.03,1)
mx2=N(nt,'ShaderNodeMix');mx2.data_type='RGBA';L.new(pup,mx2.inputs['Factor']);L.new(mx.outputs[2],mx2.inputs[6]);mx2.inputs[7].default_value=(0.004,0.003,0.003,1)
L.new(mx2.outputs[2],P.inputs['Base Color'])
P.inputs['Roughness'].default_value=0.45;P.inputs['Coat Weight'].default_value=1.0;P.inputs['Coat Roughness'].default_value=0.02
for k in range(2):
    c,o,up,bk=[Vector(E[k][3*i:3*i+3]) for i in range(4)]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96,ring_count=48,radius=1.0)
    eo=bpy.context.object;eo.name='Eye.'+('R' if k==0 else 'L')
    gz=(o+Vector((-0.2,0,-0.78))).normalized();wz=Vector((0,0,1))
    eup=(wz-gz*gz.dot(wz)).normalized();ex=eup.cross(gz).normalized()
    M=Matrix((ex,eup,gz)).transposed().to_4x4();M.translation=c
    eo.matrix_world=M@Matrix.Scale(0.302,4)
    eo.data.polygons.foreach_set('use_smooth',[True]*len(eo.data.polygons));eo.data.materials.append(em)
# tear on the camera-side eye (k=0, y<0)
fnm=np.cross(V[Fc[:,1]]-V[Fc[:,0]],V[Fc[:,2]]-V[Fc[:,0]]);VN=np.zeros_like(V)
for i in range(3): np.add.at(VN,Fc[:,i],fnm)
VN/=np.linalg.norm(VN,axis=1,keepdims=True)+1e-9
def surf(pt,off):
    j=int(np.argmin(np.linalg.norm(V-np.array(pt),axis=1)));return Vector(V[j].tolist())+Vector(VN[j].tolist())*off,Vector(VN[j].tolist())
c,o,up,bk=[Vector(E[0][3*i:3*i+3]) for i in range(4)]
p,nrm=surf(c-bk*0.14-up*0.48+o*0.2,0.03)
bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,radius=0.045)
t=bpy.context.object;t.name='Tear'
for v in t.data.vertices:
    zz=v.co.z/0.045
    if zz>0: f=1-0.75*zz**1.2;v.co.x*=f;v.co.y*=f;v.co.z*=1.6
    v.co.y*=0.6
t.data.polygons.foreach_set('use_smooth',[True]*len(t.data.polygons))
z_=Vector((0,0,1));xax=(-nrm).cross(z_).normalized();zax=xax.cross(-nrm).normalized()
t.matrix_world=Matrix.Translation(p)@Matrix((xax,-nrm,zax)).transposed().to_4x4()
tm=bpy.data.materials.new('Tear');tm.use_nodes=True;P=tm.node_tree.nodes['Principled BSDF']
P.inputs['Transmission Weight'].default_value=1;P.inputs['Roughness'].default_value=0.0;P.inputs['IOR'].default_value=1.33;P.inputs['Base Color'].default_value=(1,1,1,1)
t.data.materials.append(tm)
cu=bpy.data.curves.new('TearTrail','CURVE');cu.dimensions='3D';cu.bevel_depth=0.011;cu.bevel_resolution=4
spl=cu.splines.new('NURBS');pts=[surf(c-bk*0.27-up*0.2+o*0.18,0.006)[0],surf(c-bk*0.22-up*0.32+o*0.2,0.008)[0],surf(c-bk*0.17-up*0.42+o*0.2,0.01)[0],p+zax*0.05]
spl.points.add(len(pts)-1)
for sp_,q in zip(spl.points,pts): sp_.co=(q.x,q.y,q.z,1)
spl.use_endpoint_u=True;spl.order_u=3
tr=bpy.data.objects.new('TearTrail',cu);sc.collection.objects.link(tr);cu.materials.append(tm)
# floor
bpy.ops.mesh.primitive_plane_add(size=300);fl=bpy.context.object;fl.name='Floor'
fm=bpy.data.materials.new('Floor');fm.use_nodes=True;P=fm.node_tree.nodes['Principled BSDF'];P.inputs['Base Color'].default_value=(0.36,0.34,0.32,1);P.inputs['Roughness'].default_value=0.75
fl.data.materials.append(fm);fl.is_shadow_catcher=True
sc.render.film_transparent=True
def look(ob,tgt): ob.rotation_euler=(Vector(tgt)-ob.location).to_track_quat('-Z','Y').to_euler()
def area(name,loc,size,energy,color):
    ld=bpy.data.lights.new(name,'AREA');ld.size=size;ld.energy=energy;ld.color=color
    o=bpy.data.objects.new(name,ld);sc.collection.objects.link(o);o.location=loc;look(o,(-1.5,0,0.5) if loc[2]>0 else (-1.5,0,2))
area('Key',(-7,-7,8),5,2200,(1,0.93,0.84));area('Fill',(5,-8,3),8,500,(0.85,0.9,1));area('Rim',(2,9,6),4,1800,(1,0.97,0.92));area('Bounce',(-3,-3,-0.6),6,120,(1,0.92,0.8))
w=bpy.data.worlds.new('World');sc.world=w;w.use_nodes=True;w.node_tree.nodes['Background'].inputs[0].default_value=(0.6,0.62,0.65,1);w.node_tree.nodes['Background'].inputs[1].default_value=0.25
cams={'Hero':((-8.6,-5.2,1.35),(-2.6,-0.3,0.55),50),'Full':((-3.5,-11.5,4.6),(0.2,-0.2,0.4),42),'Face':((-6.5,-3.3,1.3),(-3.95,-0.4,0.62),52)}
for nm,(loc,tg,lens) in cams.items():
    cd=bpy.data.cameras.new(nm);cd.lens=lens;cd.dof.use_dof=nm!='Full';cd.dof.aperture_fstop=2.8 if nm=='Face' else 4
    co=bpy.data.objects.new('Cam_'+nm,cd);sc.collection.objects.link(co);co.location=loc;look(co,tg)
    cd.dof.focus_distance=(Vector(tg)-Vector(loc)).length if nm!='Hero' else (Vector(E[0][:3].tolist())-Vector(loc)).length
sc.camera=bpy.data.objects['Cam_Hero']
sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=128;sc.cycles.use_denoising=True;sc.cycles.use_adaptive_sampling=True
sc.cycles.max_bounces=8;sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast'
sc.render.resolution_x=1280;sc.render.resolution_y=720
bpy.ops.wm.save_as_mainfile(filepath='/home/claude/gecko/sad_gecko.blend')
print('saved')
