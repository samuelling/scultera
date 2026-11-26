"""Take 2, step 2: carve a visual hull of the head from 358 segmentation masks, in head coordinates."""
import numpy as np
from scipy import ndimage as ndi
d=np.load("shape_s.npz"); S=d["S"]; R=d["R"]; tr=d["tr"]; f=float(d["f"]); cx=float(d["cx"]); cy=float(d["cy"])
m=np.load('masks.npy'); T,H,W=m.shape
bg=(m==0)
# only trust background a few px away from the person (segmentation edges are soft)
bg=np.array([ndi.binary_erosion(b,iterations=2) for b in bg])
h=0.25; xs=np.arange(-11,11+h,h); ys=np.arange(-24,15+h,h); zs=np.arange(-20,12+h,h)
G=np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
votes=np.zeros(len(G),np.int16); seen=np.zeros(len(G),np.int16)
for t in range(0,T,2):
    X=G@R[t].T.astype(np.float32)+tr[t].astype(np.float32)
    u=np.round(f*X[:,0]/X[:,2]+cx).astype(int); v=np.round(f*X[:,1]/X[:,2]+cy).astype(int)
    inside=(u>=0)&(u<W)&(v>=0)&(v<H)
    seen+=inside
    votes[inside]+=bg[t][v[inside],u[inside]]
occ=(votes<=2).reshape(len(xs),len(ys),len(zs))      # carved where >2 frames saw background through it
np.savez_compressed('hull.npz',occ=occ,xs=xs,ys=ys,zs=zs,seen=seen.reshape(occ.shape))
# report silhouette widths at a few heights (just in front of the ears, z = ear line)
ear=(S[234]+S[454])/2
for y in [12,9,6,3,0,-4,-8,-12,-16]:
    j=np.argmin(abs(ys-y)); k=np.argmin(abs(zs-ear[2])); row=occ[:,j,k]
    w=(xs[row].max()-xs[row].min()) if row.any() else 0
    top=ys[occ[:,:,k].any(0)].max()
    print('y=%5.1f width %.1f cm'%(y,w))
print('top of hull at ear-plane z: %.1f   lm10 y %.1f  ear landmarks x %.1f %.1f  z %.1f'%(top,S[10,1],S[234,0],S[454,0],ear[2]))
