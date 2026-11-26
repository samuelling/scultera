"""Trim the back of the tracked-frame hull with the back-view frames (posed by silhouette), behind the ears only."""
import numpy as np
from scipy import ndimage as ndi
h=np.load('hull_t2.npz'); occ=h['occ'].copy(); xs,ys,zs=h['xs'],h['ys'],h['zs']
d=np.load('poses_sil.npz'); sel=d['src']=='silhouette'; R,tr,fr=d['R'][sel],d['tr'][sel],d['frame'][sel]
sh=np.load('shape_s.npz'); f=float(sh['f']); cx=float(sh['cx']); cy=float(sh['cy'])
masks=np.load('masks.npy',mmap_mode='r'); H,W=masks.shape[1:]
I=np.argwhere(occ); P=np.c_[xs[I[:,0]],ys[I[:,1]],zs[I[:,2]]].astype(np.float32)
back=(P[:,2]<-9.5)&(P[:,1]>-12)
Ib,Pb=I[back],P[back]; bgv=np.zeros(len(Pb),np.int16)
for k in range(0,len(fr),2):
    m=np.asarray(masks[fr[k]]); bg=ndi.binary_erosion(m==0,iterations=3)
    X=Pb@R[k].T.astype(np.float32)+tr[k].astype(np.float32)
    u=np.round(f*X[:,0]/X[:,2]+cx).astype(int); v=np.round(f*X[:,1]/X[:,2]+cy).astype(int)
    ok=(u>=0)&(u<W)&(v>=0)&(v<H); bgv[ok]+=bg[v[ok],u[ok]]
cut=bgv>8
occ[tuple(Ib[cut].T)]=False
print('back trim removed %d of %d voxels behind the ears'%(cut.sum(),len(Pb)))
np.savez_compressed('hull_t2b.npz',occ=occ,xs=xs,ys=ys,zs=zs)
