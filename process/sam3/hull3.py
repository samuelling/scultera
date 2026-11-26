"""Visual hull of the head from posed frames (tracked + silhouette-solved). Head classes: hair, face skin (+ body skin above the jaw)."""
import numpy as np, sys
from scipy import ndimage as ndi
d=np.load(sys.argv[1] if len(sys.argv)>1 else 'poses_all.npz')
R,tr,fr=d['R'],d['tr'],d['frame']; sh=np.load('shape_s.npz'); f=float(sh['f']); cx=float(sh['cx']); cy=float(sh['cy'])
masks=np.load('masks.npy',mmap_mode='r'); H,W=masks.shape[1:]
h=float(sys.argv[2]) if len(sys.argv)>2 else 0.25
xs=np.arange(-12,12+h,h); ys=np.arange(-26,16+h,h); zs=np.arange(-22,14+h,h)
G=np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
bgv=np.zeros(len(G),np.int16); seen=np.zeros(len(G),np.int16)
for k in range(len(fr)):
    m=np.asarray(masks[fr[k]]); bg=ndi.binary_erosion(m==0,iterations=2)
    X=G@R[k].T.astype(np.float32)+tr[k].astype(np.float32)
    u=np.round(f*X[:,0]/X[:,2]+cx).astype(np.int32); v=np.round(f*X[:,1]/X[:,2]+cy).astype(np.int32)
    ins=(u>=0)&(u<W)&(v>=0)&(v<H)
    seen+=ins; bgv[ins]+=bg[v[ins],u[ins]]
occ=(bgv<=int(sys.argv[4]) if len(sys.argv)>4 else bgv<=3).reshape(len(xs),len(ys),len(zs))
np.savez_compressed(sys.argv[3] if len(sys.argv)>3 else 'hull3.npz',occ=occ,xs=xs,ys=ys,zs=zs)
print('hull voxels %d, frames %d'%(occ.sum(),len(fr)))
