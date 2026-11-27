"""Distance refit from silhouette size, only where he leans (not the two clean chair turns, whose interpolated distance checks out)."""
import numpy as np, sys
sys.path.insert(0,'proc'); from common import *
from scipy import ndimage as ndi
p=np.load('poses4.npz'); R,tr,src=p['R'],p['tr'].copy(),p['src']; masks=np.load(B3+'masks.npy',mmap_mode='r'); N,H,W=masks.shape
h=np.load('hull4.npz'); I=np.argwhere(h['occ_t']&~ndi.binary_erosion(h['occ_t'])); Pm=np.c_[h['xs'][I[:,0]],h['ys'][I[:,1]],h['zs'][I[:,2]]]; Pm=Pm[tonew(Pm)[:,1]>-7]; c_head=Pm.mean(0)
def sil(Rm,t3):
    X=Pm@Rm.T+t3; u=np.round((F0*X[:,0]/X[:,2]+CX)/4).astype(int); v=np.round((F0*X[:,1]/X[:,2]+CY)/4).astype(int)
    s=np.zeros((H//4,W//4),bool); ok=(u>=0)&(u<W//4)&(v>=0)&(v<H//4); s[v[ok],u[ok]]=True; return ndi.binary_closing(s,iterations=2),int(v.max())
n=0
for t in np.where(src=='solved')[0]:
    if 105<=t<298 or 409<=t<550: continue
    mk=np.isin(np.asarray(masks[t])[::4,::4],[1,3])
    for it in range(3):
        s,vmax=sil(R[t],tr[t]); m=mk.copy(); m[vmax:]=False
        if m.sum()<80 or s.sum()<80: break
        Z=np.clip(tr[t][2]*np.sqrt(s.sum()/m.sum()),50,140); yy,xx=np.nonzero(m); ys_,xs_=np.nonzero(s)
        c=R[t]@c_head+tr[t]; c=c*Z/c[2]; c[:2]+=(np.array([xx.mean(),yy.mean()])-np.array([xs_.mean(),ys_.mean()]))*4*Z/F0; tr[t]=c-R[t]@c_head
    n+=1
np.savez('poses4.npz',R=R,tr=tr,src=src,n=p['n'],kbest=p['kbest']); print('refit distance on',n,'leaning frames;',(src=='solved').sum(),'solved,',(src=='tracked').sum(),'tracked')
