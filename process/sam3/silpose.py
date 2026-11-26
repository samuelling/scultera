"""Poses for the frames where no face is visible (back of the head): a yaw path about the camera's vertical axis,
chosen by silhouette agreement with the hull from tracked frames, smooth in time (dynamic programming),
pinned to the tracked poses on both sides of each gap."""
import numpy as np, sys
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as Rot
sh=np.load('shape_s.npz'); f=float(sh['f']); cx=float(sh['cx']); cy=float(sh['cy'])
good=sh['good']; idx=sh['idx']; Rg=sh['R'][good]; tg=sh['tr'][good]; fg=idx[good]
masks=np.load('masks.npy',mmap_mode='r'); N=masks.shape[0]; DS=4; H,W=masks.shape[1]//DS,masks.shape[2]//DS
hl=np.load(sys.argv[1] if len(sys.argv)>1 else 'hull_tracked.npz'); occ=hl['occ']; xs,ys,zs=hl['xs'],hl['ys'],hl['zs']
I=np.argwhere(occ); P=np.c_[xs[I[:,0]],ys[I[:,1]],zs[I[:,2]]]; P=P[P[:,1]>-10.0]
rng=np.random.default_rng(0); P=P[rng.choice(len(P),min(len(P),20000),replace=False)]
c_head=P.mean(0)
def headmask(t):
    m=np.asarray(masks[t])[::DS,::DS]; return np.isin(m,[1,3])
def sil(Rm,t3):
    X=P@Rm.T+t3; u=(f*X[:,0]/X[:,2]+cx)/DS; v=(f*X[:,1]/X[:,2]+cy)/DS
    ui=np.round(u).astype(int); vi=np.round(v).astype(int); ok=(ui>=0)&(ui<W)&(vi>=0)&(vi<H)
    s=np.zeros((H,W),bool); s[vi[ok],ui[ok]]=True; return ndi.binary_closing(s,iterations=2)
def place(Rm,mk,Z):
    """translation so that the head centroid projects onto the mask centroid at depth Z"""
    yy,xx=np.nonzero(mk)
    if len(xx)<20: return None
    u=xx.mean()*DS; v=yy.mean()*DS
    tgt=np.array([(u-cx)*Z/f,(v-cy)*Z/f,Z]); return tgt-Rm@c_head
def Ry(a):
    a=np.radians(a); return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
gset=set(fg.tolist()); pos={fr:i for i,fr in enumerate(fg)}
out_R={fr:Rg[pos[fr]] for fr in fg}; out_t={fr:tg[pos[fr]] for fr in fg}; src={fr:'tracked' for fr in fg}
gaps=[]; t=0
while t<N:
    if t in gset: t+=1; continue
    s=t
    while t<N and t not in gset: t+=1
    gaps.append((s,t))
print('gaps:',[(a/30,b/30) for a,b in gaps if b-a>3])
TH=np.arange(-240,241,4.0)
for a,b in gaps:
    if a==0 or b>=N: continue
    pa,pb=a-1,b; R0=out_R[pa]; Z0=out_t[pa][2]; Z1=out_t[pb][2]
    # candidate end angles: rotation about camera y that best maps start to end
    Rd=out_R[pb]@R0.T; th_end=np.degrees(np.arctan2(Rd[0,2],Rd[2,2]))
    n=b-a; costs=np.zeros((n,len(TH)))
    for k in range(n):
        t=a+k; mk=headmask(t); Z=Z0+(Z1-Z0)*(k+1)/(n+1)
        for j,th in enumerate(TH):
            Rm=Ry(th)@R0; t3=place(Rm,mk,Z)
            if t3 is None: costs[k,j]=1; continue
            s=sil(Rm,t3); inter=(s&mk).sum(); uni=(s|mk).sum(); costs[k,j]=1-inter/max(uni,1)
    # DP: start at 0 (frame a-1), end at th_end (+-360)
    lam=0.004
    trans=lam*((TH[:,None]-TH[None,:])/4.0)**2
    acc=costs[0]+lam*(TH/4.0)**2; back=np.zeros((n,len(TH)),int)
    for k in range(1,n):
        m=acc[:,None]+trans; back[k]=m.argmin(0); acc=m.min(0)+costs[k]
    endc=np.min([lam*((TH-(th_end+o))/4.0)**2 for o in (-360,0,360)],0)
    j=int(np.argmin(acc+endc)); path=[j]
    for k in range(n-1,0,-1): j=back[k,j]; path.append(j)
    path=path[::-1]; ths=TH[path]
    for k in range(n):
        t=a+k; Rm=Ry(ths[k])@R0; mk=headmask(t); Z=Z0+(Z1-Z0)*(k+1)/(n+1); t3=place(Rm,mk,Z)
        if t3 is None: continue
        out_R[t]=Rm; out_t[t]=t3; src[t]='silhouette'
    if n>3: print('gap %.1f-%.1fs: yaw path %+.0f .. %+.0f .. %+.0f (end target %+.0f), mean IoU %.2f'%(a/30,b/30,ths[0],ths[np.argmax(np.abs(ths))],ths[-1],th_end,1-np.mean(costs[np.arange(n),path])))
frs=np.array(sorted(out_R)); np.savez('poses_all.npz',R=np.array([out_R[k] for k in frs]),tr=np.array([out_t[k] for k in frs]),frame=frs,src=np.array([src[k] for k in frs]))
print('posed frames',len(frs),'of',N)
