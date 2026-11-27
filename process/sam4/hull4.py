"""Take 4, step 2: fix the solved frames' distance from silhouette size, sharpen the masks on image edges, re-carve with the back views."""
import numpy as np, cv2, sys
sys.path.insert(0,'proc'); from common import *
from scipy import ndimage as ndi
p=np.load('poses4.npz'); R,tr,src=p['R'],p['tr'].copy(),p['src']
masks=np.load(B3+'masks.npy',mmap_mode='r'); N,H,W=masks.shape
fin=np.load(B3+'finished.npz'); V=fin['v'].astype(float); rng=np.random.default_rng(0)
Pm=toold(V[V[:,1]>-7.0]); Pm=Pm[rng.choice(len(Pm),30000,replace=False)]; c_head=Pm.mean(0)
def sil(Rm,t3):
    X=Pm@Rm.T+t3; u=np.round((F0*X[:,0]/X[:,2]+CX)/4).astype(int); v=np.round((F0*X[:,1]/X[:,2]+CY)/4).astype(int)
    s=np.zeros((H//4,W//4),bool); ok=(u>=0)&(u<W//4)&(v>=0)&(v<H//4); s[v[ok],u[ok]]=True; return ndi.binary_closing(s,iterations=2),int(v.max())
solved=np.where(src=='solved')[0]; zfix=[]
for t in solved:
    mk=np.isin(np.asarray(masks[t])[::4,::4],[1,3])
    for it in range(3):
        s,vmax=sil(R[t],tr[t]); m=mk.copy(); m[vmax:]=False
        if m.sum()<50 or s.sum()<50: break
        Z=tr[t][2]*np.sqrt(s.sum()/m.sum()); yy,xx=np.nonzero(m); cm=np.array([xx.mean()*4,yy.mean()*4])
        ys_,xs_=np.nonzero(s); cs=np.array([xs_.mean()*4,ys_.mean()*4])
        c=R[t]@c_head+tr[t]; c=c*Z/c[2]; c[:2]+=(cm-cs)*Z/F0; tr[t]=c-R[t]@c_head
    zfix.append(tr[t][2])
print('solved frames: distance refit from silhouette size, now %.0f-%.0f cm'%(min(zfix),max(zfix)))
np.savez('poses4.npz',R=R,tr=tr,src=src,n=p['n'],kbest=p['kbest'])
def gf_mask(t,r=7,eps=2e-3):
    """guided filter: snap the soft person mask to edges of the image"""
    I=cv2.imread(B3+'frames/f%04d.jpg'%(t+1),0).astype(np.float32)/255; pm=(np.asarray(masks[t])!=0).astype(np.float32)
    k=(2*r+1,2*r+1); mI=cv2.blur(I,k); mp=cv2.blur(pm,k); a=(cv2.blur(I*pm,k)-mI*mp)/(cv2.blur(I*I,k)-mI*mI+eps); b=mp-a*mI
    return cv2.blur(a,k)*I+cv2.blur(b,k)
h=0.2; xs=np.arange(-12,12+h,h); ys=np.arange(-18,16+h,h); zs=np.arange(-22,14+h,h)
G=np.stack(np.meshgrid(xs,ys,zs,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
tracked=np.where(src=='tracked')[0]; back=np.r_[np.arange(105,298,2),np.arange(409,550,2)]
votes={}
for name,frs in (('tracked',tracked),('back',back)):
    bg=np.zeros(len(G),np.int16)
    for t in frs:
        m=gf_mask(t)<0.5; m=ndi.binary_erosion(m,iterations=1 if name=='tracked' else 3)
        X=G@R[t].T.astype(np.float32)+tr[t].astype(np.float32); u=np.round(F0*X[:,0]/X[:,2]+CX).astype(np.int32); v=np.round(F0*X[:,1]/X[:,2]+CY).astype(np.int32)
        ok=(u>=0)&(u<W)&(v>=0)&(v<H); bg[ok]+=m[v[ok],u[ok]]
    votes[name]=bg.reshape(len(xs),len(ys),len(zs)); print(name,len(frs),'frames carved')
occ_t=votes['tracked']<=3; occ=occ_t&(votes['back']<=6)
np.savez_compressed('hull4.npz',occ=occ,occ_t=occ_t,xs=xs,ys=ys,zs=zs)
print('voxels: tracked-only hull %d -> with back views %d (%.0f%% removed)'%(occ_t.sum(),occ.sum(),100*(1-occ.sum()/occ_t.sum())))
