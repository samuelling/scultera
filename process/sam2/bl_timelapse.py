"""Render the take-2 time-lapse: measurement stills, blockout stages, details, finish, paint, turntable, performance."""
import sys, os, json, math; sys.path.insert(0,'proc')
import bpy, numpy as np
from bl_common import *
from scipy import sparse as sp
from scipy.spatial.transform import Rotation as Rot
OUT='tl'; RES=int(os.environ.get('RES','560')); SPP=int(os.environ.get('SPP','12'))
ONLY=os.environ.get('ONLY')
sc=reset(); studio(sc); cycles(sc,RES,RES,SPP)
TGT=np.array([0,-4.5,-5.5]); FT=(0,-1.0,-2.5)
def cam_at(theta,elev=4.0,dist=78,lens=62,tgt=None):
    for o in list(bpy.data.objects):
        if o.type=='CAMERA': bpy.data.objects.remove(o)
    T_=TGT if tgt is None else np.array(tgt); th=math.radians(theta); p=T_+np.array([dist*math.sin(th),elev,dist*math.cos(th)])
    tb=(T_[0],-T_[2],T_[1]); pb=(p[0],-p[2],p[1]); camera(sc,pb,tb,lens)
def clear():
    for o in list(bpy.data.objects):
        if o.type=='MESH': bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes): bpy.data.meshes.remove(m)
def rnd(name):
    sc.render.filepath=os.path.abspath(f'{OUT}/{name}.png'); bpy.ops.render.render(write_still=True); print('rendered',name,flush=True)
def spheres(P,r=0.13,col=(0.85,0.45,0.15)):
    t=np.linspace(0,np.pi,6); p=np.linspace(0,2*np.pi,8,endpoint=False); T,Pp=np.meshgrid(t,p,indexing='ij')
    u=np.stack([np.sin(T)*np.cos(Pp),np.cos(T),np.sin(T)*np.sin(Pp)],-1).reshape(-1,3); nv=len(u)
    Fu=[]
    for i in range(5):
        for j in range(8): a=i*8+j;b=i*8+(j+1)%8;c=(i+1)*8+j;d=(i+1)*8+(j+1)%8;Fu+=[[a,c,b],[b,c,d]]
    Fu=np.array(Fu); V=(P[:,None,:]+u[None]*r).reshape(-1,3); F=(Fu[None]+nv*np.arange(len(P))[:,None,None]).reshape(-1,3)
    add_mesh('pts',V,F,[mk_mat('dot',col,0.4)])
clay=lambda: mk_mat('clay',CLAY,0.5,0.15)
fc=np.load('face.npz'); S=fc['S']; D=fc['D']; fm=np.load('rmap_face.npz')
stills=[]   # (name, theta)
def want(n): return ONLY is None or n.startswith(ONLY)
th=iter(np.linspace(-38,22,24))
# --- measurement: one raw frame of tracked points, then the average
if want('s01'):
    clear(); cam_at(next(th),2,52,62,FT); spheres(S[:468]+D[0,:468]); rnd('s01_points_frame1')
else: next(th)
if want('s02'):
    clear(); cam_at(next(th),2,52,62,FT); spheres(S[:468]+np.median(D[:30,:468],0)); rnd('s02_points_30')
else: next(th)
if want('s03'):
    clear(); cam_at(next(th),2,52,62,FT); spheres(S[:468]); rnd('s03_points_all')
else: next(th)
if want('s04'):
    clear(); cam_at(next(th),2,52,62,FT); spheres(S[:468],0.09,(0.6,0.35,0.15)); add_mesh('face',fm['Vs'],fm['Fs'],[clay()]); rnd('s04_face_surface')
else: next(th)
for nm,fn in [('s05_skull','mesh_skull.npz'),('s06_face_on_skull','mesh_face.npz'),('s07_neck_bust','mesh_neck.npz'),('s08_ears','mesh_ears.npz'),
              ('s09_nostrils','mesh_d1.npz'),('s10_lids','mesh_d2.npz'),('s11_brows','mesh_d3.npz'),('s12_lip_border','mesh_final.npz')]:
    t_=next(th)
    if not want(nm[:3]): continue
    clear(); cam_at(t_,*((2,60,62,FT) if nm in ('s05_skull','s06_face_on_skull') else ())); m=np.load(fn); add_mesh('head',m['v'],m['f'],[clay()])
    if nm=='s05_skull': add_mesh('face',fm['Vs'],fm['Fs'],[mk_mat('face',(0.75,0.55,0.4),0.5)])
    rnd(nm)
fin=np.load('finished.npz')
def finish_scene(stage):
    d=fin; F=d['f']; mat=d['mat']
    from scipy import stats
    fmode=stats.mode(mat[F],axis=1).mode.ravel()
    V=np.load('mesh_final.npz')['v'] if stage<2 else d['v']
    mats=[clay(),mk_mat('hair',HAIR,0.8),mk_mat('shirt',SHIRT,0.85),mk_mat('mouth',MOUTH,0.6)]
    idx=np.where(fmode==3,3,np.where(fmode==1,1 if stage>=2 else 0,np.where(fmode==2,2 if stage>=3 else 0,0)))
    if stage<2: add_mesh('head',V,np.load('mesh_final.npz')['f'],[clay()])
    else: add_mesh('head',V,F,mats,idx.astype(int))
    em=mk_mat('eye',EYE,0.35,0.05); tm=mk_mat('teeth',TOOTH,0.35,0.1)
    add_mesh('eyeR',d['eyeR_v'],d['eyeR_f'],[em]); add_mesh('eyeL',d['eyeL_v'],d['eyeL_f'],[em])
    add_mesh('teethU',d['tu_v'],d['tu_f'],[tm]); add_mesh('teethL',d['tl_v'],d['tl_f'],[tm])
for nm,stage in [('s13_eyes_teeth',1),('s14_hair',2),('s15_shirt',3)]:
    t_=next(th)
    if not want(nm[:3]): continue
    clear(); cam_at(t_); finish_scene(stage); rnd(nm)
# paint fade-in: blend clay colour -> paint colour on the vertex colours
P=np.load('paint.npy'); mat=fin['mat']
claycol=np.array([CLAY,HAIR,SHIRT,MOUTH,(0.25,0.17,0.14)])[mat]
for nm,a in [('s16_paint_1',0.35),('s17_paint_2',0.7),('s18_paint_3',1.0)]:
    t_=next(th)
    if not want(nm[:3]): continue
    clear(); cam_at(t_); objs=load_painted()
    set_colors(objs[0],claycol*(1-a)+P*a,'paint2'); objs[0].data.materials[0].node_tree.nodes['Attribute'].attribute_name='paint2'
    rnd(nm)
# turntable (painted, rest)
if want('turn'):
    clear(); objs=load_painted()
    for i,a in enumerate(np.linspace(22,382,48,endpoint=False)):
        cam_at(a); rnd('turn_%03d'%i)
# performance: the 12 s clip replayed on the sculpt (every 3rd video frame = 10 fps)
if want('perf'):
    clear(); objs=load_painted(); head,eR,eL,tu,tl=objs
    Wv=sp.load_npz('Wv.npz'); an=np.load('anim.npz'); V0=fin['v'].astype(float)
    piv=np.array([0,-10.0,-8.0])
    def hw(V):
        x=np.clip((V[:,1]+19)/7,0,1); return x*x*(3-2*x)
    W0=hw(V0); Wt=1.0
    def bl(V): return np.c_[V[:,0],-V[:,2],V[:,1]]
    cam_at(-8,elev=3,dist=70)
    for k,t in enumerate(range(0,358,3)[:int(os.environ.get('PERFN','999'))]):
        Q=Rot.from_quat(an['quat'][t]).as_matrix()
        def pose(V,w): return V*(1-w[:,None])+((V-piv)@Q.T+piv)*w[:,None]
        Vh=pose(V0+Wv@D[t,:468],W0)
        head.data.vertices.foreach_set('co',bl(Vh).ravel()); head.data.update()
        for ob,key,c,g in ((eR,'eyeR_v','eyeR_c',an['gaze'][t,0]),(eL,'eyeL_v','eyeL_c',an['gaze'][t,1])):
            Rg=Rot.from_euler('yx',[g[0],-g[1]],degrees=True).as_matrix(); Ve=(fin[key]-fin[c])@Rg.T+fin[c]
            ob.data.vertices.foreach_set('co',bl(pose(Ve,hw(Ve))).ravel()); ob.data.update()
        tu.data.vertices.foreach_set('co',bl(pose(fin['tu_v'].astype(float),hw(fin['tu_v']))).ravel()); tu.data.update()
        tl.data.vertices.foreach_set('co',bl(pose(fin['tl_v']+an['jaw'][t],hw(fin['tl_v']))).ravel()); tl.data.update()
        rnd('perf_%03d'%k)
