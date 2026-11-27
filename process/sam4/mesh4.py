"""Take 4, step 4: marching cubes of the signed-distance field (final + the stages shown in the time-lapse)."""
import numpy as np, sys, time
sys.path.insert(0,'proc'); import sdf4
from skimage import measure
def cube(parts,H,name):
    gx=np.arange(-22,22+H,H); gy=np.arange(-31,14+H,H); gz=np.arange(-23,9+H,H); vol=np.empty((len(gx),len(gy),len(gz)),np.float32)
    Y,Z=np.meshgrid(gy,gz,indexing='ij')
    for i,x in enumerate(gx): vol[i]=sdf4.sdf(np.stack([np.full_like(Y,x),Y,Z],-1),parts)
    v,f,_,_=measure.marching_cubes(vol,0,spacing=(H,H,H)); v+=[gx[0],gy[0],gz[0]]
    np.savez_compressed('mesh4_%s.npz'%name,v=v.astype(np.float32),f=f); print(name,len(v),'verts',flush=True)
for parts,name in [(('raw',),'s1_carve'),(('head',),'s2_smoothed'),(('head','face'),'s3_face'),(('head','face','neck','bust'),'s4_bust')]: cube(parts,0.2,name)
cube(('head','face','neck','bust','details'),0.15,'final')
