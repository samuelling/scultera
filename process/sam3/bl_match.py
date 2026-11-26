"""Render the sculpt through the tracked camera of given video frames, side by side with the frame (local check only)."""
import sys; sys.path.insert(0,'proc')
import bpy, numpy as np
from bl_common import *
from PIL import Image
out=sys.argv[1]; frames=[int(x) for x in sys.argv[2].split(',')]; spp=int(sys.argv[3]) if len(sys.argv)>3 else 10; DEFORM=len(sys.argv)>4 and 'deform' in sys.argv[4]; PAINT=len(sys.argv)>4 and 'paint' in sys.argv[4]
fc=np.load('face.npz'); d=np.load('shape_s.npz'); R,tr,f=d['R'],d['tr'],float(d['f'])
fin=np.load('finished.npz')
from scipy import sparse as sp
from scipy.spatial.transform import Rotation as Rot
Wv=sp.load_npz('Wv.npz'); D=fc['D']; an=np.load('anim.npz')
def posed(t):
    out={'v':fin['v']+Wv@D[t,:468],'tu_v':fin['tu_v'],'tl_v':fin['tl_v']+an['jaw'][t]}
    for k,c,g in (('eyeR_v','eyeR_c',an['gaze'][t,0]),('eyeL_v','eyeL_c',an['gaze'][t,1])):
        Rg=Rot.from_euler('yx',[g[0],-g[1]],degrees=True).as_matrix(); out[k]=(fin[k]-fin[c])@Rg.T+fin[c]
    return out
def to_cam(V,t):
    vo=(V+np.array([fc['shx'],0,0])-fc['m0'])@fc['Rh']+fc['m0']; X=vo@R[t].T+tr[t]
    return np.c_[X[:,0],-X[:,1],-X[:,2]]           # opencv -> blender camera space (already "blender-ish" xyz)
pairs=[]
for t in frames:
    sc=reset()
    objs=(load_painted() if PAINT else load_finished()); PO=posed(t) if DEFORM else {k:fin[k] for k in ['v','eyeR_v','eyeL_v','tu_v','tl_v']}
    # replace geometry by camera-space coords: add_mesh converts (x,y,z)->(x,-z,y); feed inverse so result = desired
    for ob,key in zip(objs,['v','eyeR_v','eyeL_v','tu_v','tl_v']):
        Xb=to_cam(PO[key].astype(float),t)          # desired blender coords
        ob.data.vertices.foreach_set('co',Xb.ravel()); ob.data.update()
    cd=bpy.data.cameras.new('cam'); cd.sensor_fit='VERTICAL'; cd.sensor_height=24; cd.lens=f*24/1280
    cam=bpy.data.objects.new('cam',cd); sc.collection.objects.link(cam); sc.camera=cam; cam.location=(0,0,0); cam.rotation_euler=(0,0,0)
    for nm,loc,e,s in (('key',(-25,25,10),9000,30),('fill',(30,-5,0),2500,40),('rim',(5,30,-60),5000,20)):
        ld=bpy.data.lights.new(nm,'AREA');ld.energy=e;ld.size=s;l=bpy.data.objects.new(nm,ld);sc.collection.objects.link(l);l.location=loc;look(l,(0,0,-44))
    w=bpy.data.worlds.new('w');sc.world=w;w.use_nodes=True;w.node_tree.nodes['Background'].inputs[0].default_value=(0.5,0.48,0.45,1)
    cycles(sc,720,1280,spp); sc.render.resolution_percentage=50; sc.render.filepath=f'/tmp/m_{t}.png'
    bpy.ops.render.render(write_still=True)
    a=Image.open(f'/tmp/m_{t}.png').convert('RGB').crop((30,75,330,450)); b=Image.open('frames/f%04d.jpg'%(t+1)).resize((360,640)).crop((30,75,330,450))
    pairs.append((b,a))
c=Image.new('RGB',(600*len(pairs)//1 if len(pairs)<3 else 1200,375*((len(pairs)+1)//2)))
for i,(b,a) in enumerate(pairs):
    x=(i%2)*600; y=(i//2)*375; c.paste(b,(x,y)); c.paste(a,(x+300,y))
c.save(out)
