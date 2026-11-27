"""Take-4 time-lapse frames: carve stages, finish, paint fade, turntable, then the first 20 s of the clip replayed (chair turns included)."""
import sys, os, math; sys.path.insert(0,'proc')
import bpy, numpy as np
from bl4 import *
from scipy import sparse as sp
from scipy.spatial.transform import Rotation as Rot
RES=int(os.environ.get('RES','560')); SPP=int(os.environ.get('SPP','12')); ONLY=os.environ.get('ONLY'); want=lambda n: ONLY is None or n.startswith(ONLY)
TGT=np.array([0,-4.5,-5.5])
def clear():
    for o in list(bpy.data.objects):
        if o.type in ('MESH','CAMERA'): bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes): bpy.data.meshes.remove(m)
def cam_at(theta,elev=4.0,dist=78,lens=62,tgt=TGT):
    for o in list(bpy.data.objects):
        if o.type=='CAMERA': bpy.data.objects.remove(o)
    th=math.radians(theta); p=np.array(tgt)+np.array([dist*math.sin(th),elev,dist*math.cos(th)]); camera(sc,(p[0],-p[2],p[1]),(tgt[0],-tgt[2],tgt[1]),lens)
def rnd(name): sc.render.filepath=os.path.abspath('tl/%s.png'%name); bpy.ops.render.render(write_still=True); print('rendered',name,flush=True)
sc=reset(); studio(sc); cycles(sc,RES,RES,SPP)
if want('cmp'):
    for mode in ('clay','paint'):
        clear(); load4(mode); camera(sc,(-14,-70,1.5),(0,0,-1.6),lens=70); rnd('cmp_'+mode)
th=iter(np.linspace(-40,25,9))
for nm,fn in [('s01_carve','mesh4_s1_carve.npz'),('s02_smoothed','mesh4_s2_smoothed.npz'),('s03_face','mesh4_s3_face.npz'),('s04_bust','mesh4_s4_bust.npz'),('s05_details','mesh4_final.npz')]:
    t_=next(th)
    if not want(nm[:3]): continue
    clear(); cam_at(t_,*((3,70,62,(0,-2,-6)) if nm<'s04' else ())); m=np.load(fn); add_mesh('m',m['v'],m['f'][:,::-1] if nm<'s05' else m['f'][:,::-1],[mk_mat('clay',(0.50,0.37,0.29),0.55,0.1)]); rnd(nm)
for nm,(mode,fade) in [('s06_finish',('clay',1)),('s07_paint_1',('paint',0.35)),('s08_paint_2',('paint',0.7)),('s09_paint_3',('paint',1.0))]:
    t_=next(th)
    if not want(nm[:3]): continue
    clear(); cam_at(t_); load4(mode,fade); rnd(nm)
if want('turn'):
    clear(); load4('paint')
    for i,a in enumerate(np.linspace(25,385,48,endpoint=False)): cam_at(a); rnd('turn_%03d'%i)
if want('perf'):
    clear(); head,eR,eL,tu,tl=load4('paint'); fin=np.load('finished4.npz'); an=np.load('anim4.npz'); Wv=sp.load_npz('Wv4.npz'); V0=fin['v'].astype(float)
    piv=np.array([0,-10.0,-8.0]); ax=np.array([0,-10.0,-8.6]); nh=an['nh']; K=int(an['K'])
    def hw(V): x=np.clip((V[:,1]+19)/7,0,1); return x*x*(3-2*x)
    bl=lambda V: np.c_[V[:,0],-V[:,2],V[:,1]]
    cam_at(-6,elev=3,dist=74)
    for k,t in enumerate(range(0,600,3)[:int(os.environ.get('PERFN','999'))]):
        Rs=Rot.from_quat(an['swing'][t]).as_matrix(); Rc=Rot.from_rotvec(nh*np.radians(an['psi'][t])).as_matrix(); Dt=np.tensordot(an['coef'][t],an['comps'],1)
        def pose(V):
            w=hw(V)[:,None]; V=V*(1-w)+((V-piv)@Rs.T+piv)*w; return (V-ax)@Rc.T+ax
        head.data.vertices.foreach_set('co',bl(pose(V0+Wv@Dt)).ravel()); head.data.update()
        for ob,key,g in ((eR,'eyeR',an['gaze'][t,0]),(eL,'eyeL',an['gaze'][t,1])):
            c=fin[key+'_c']; Rg=Rot.from_euler('yx',[g[0],-g[1]],degrees=True).as_matrix(); ob.data.vertices.foreach_set('co',bl(pose((fin[key+'_v']-c)@Rg.T+c)).ravel()); ob.data.update()
        tu.data.vertices.foreach_set('co',bl(pose(fin['tu_v'].astype(float))).ravel()); tu.data.update()
        tl.data.vertices.foreach_set('co',bl(pose(fin['tl_v']+an['jaw'][t])).ravel()); tl.data.update(); rnd('perf_%03d'%k)
