"""Poses through the back of the head by model-based visual odometry: track features on the hair/skin from frame to
frame (KLT), lift them onto the current head model, solve PnP, chain; then close the loop on the tracked pose
at the far side of each gap and spread the drift over the gap."""
import numpy as np, sys, cv2
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as Rot, Slerp
sh=np.load('shape_s.npz'); f=float(sh['f']); cx=float(sh['cx']); cy=float(sh['cy'])
good=sh['good']; idx=sh['idx']; Rg=sh['R'][good]; tg=sh['tr'][good]; fg=idx[good]
masks=np.load('masks.npy',mmap_mode='r'); N,H,W=masks.shape
hl=np.load(sys.argv[1] if len(sys.argv)>1 else 'hull_tracked.npz'); occ=hl['occ']; xs,ys,zs=hl['xs'],hl['ys'],hl['zs']
# surface points of the model (boundary voxels)
er=ndi.binary_erosion(occ); I=np.argwhere(occ&~er); P=np.c_[xs[I[:,0]],ys[I[:,1]],zs[I[:,2]]].astype(np.float32); P=P[P[:,1]>-12]
K=np.array([[f,0,cx],[0,f,cy],[0,0,1]],float)
def gray(t): return cv2.cvtColor(cv2.imread('frames/f%04d.jpg'%(t+1)),cv2.COLOR_BGR2GRAY)
def depthmap(Rm,t3):
    X=P@Rm.T+t3; u=np.round(f*X[:,0]/X[:,2]+cx).astype(int); v=np.round(f*X[:,1]/X[:,2]+cy).astype(int)
    ok=(u>=1)&(u<W-1)&(v>=1)&(v<H-1); zb=np.full((H,W),np.inf); idxb=np.full((H,W),-1)
    o=np.argsort(-X[ok,2]); uu,vv,zz,ii=u[ok][o],v[ok][o],X[ok,2][o],np.where(ok)[0][o]
    for du in (-1,0,1):
        for dv in (-1,0,1):
            zb[vv+dv,uu+du]=zz; idxb[vv+dv,uu+du]=ii   # nearer written last
    return zb,idxb
c_head=P[P[:,1]>-10].mean(0)
gset=set(fg.tolist()); pos={fr:i for i,fr in enumerate(fg)}
outR={fr:Rg[pos[fr]] for fr in fg}; outT={fr:tg[pos[fr]] for fr in fg}; src={fr:'tracked' for fr in fg}
gaps=[]; t=0
while t<N:
    if t in gset: t+=1; continue
    s=t
    while t<N and t not in gset: t+=1
    gaps.append((s,t))
for a,b in gaps:
    if a==0 or b>=N: continue
    Rm,t3=outR[a-1],outT[a-1].copy(); chainR=[];chainT=[]; prev=gray(a-1); lost=False
    for t in range(a,b+1):
        cur=gray(t); hm=np.isin(np.asarray(masks[t-1]),[1,3]); hm=ndi.binary_erosion(hm,iterations=14)
        zb,ib=depthmap(Rm,t3)
        pts=cv2.goodFeaturesToTrack(prev,600,0.002,4,mask=(hm&np.isfinite(zb)).astype(np.uint8))
        if pts is None or len(pts)<12: lost=True; chainR.append(Rm); chainT.append(t3); prev=cur; continue
        nxt,st,_=cv2.calcOpticalFlowPyrLK(prev,cur,pts,None,winSize=(13,13),maxLevel=3)
        back,st2,_=cv2.calcOpticalFlowPyrLK(cur,prev,nxt,None,winSize=(13,13),maxLevel=3)
        okk=(st.ravel()==1)&(st2.ravel()==1)&(np.linalg.norm(back-pts,axis=2).ravel()<1.0)
        p0=pts[okk,0]; p1=nxt[okk,0]
        pi=ib[p0[:,1].astype(int),p0[:,0].astype(int)]; keep=pi>=0
        if keep.sum()<10: lost=True; chainR.append(Rm); chainT.append(t3); prev=cur; continue
        obj=P[pi[keep]].astype(np.float64); img=p1[keep].astype(np.float64)
        # rotation only; translation pinned by the head silhouette's centroid (removes the turn-vs-slide ambiguity)
        mk=np.isin(np.asarray(masks[t]),[1,3]); yy,xx=np.nonzero(mk[::2,::2]); uc=xx.mean()*2; vc=yy.mean()*2
        Zc=(Rm@c_head+t3)[2]
        def pose_of(rv):
            Rn=Rot.from_rotvec(rv).as_matrix(); cc=np.array([(uc-cx)*Zc/f,(vc-cy)*Zc/f,Zc]); return Rn,cc-Rn@c_head
        def res(rv):
            Rn,tn=pose_of(rv); X=obj@Rn.T+tn; return np.r_[f*X[:,0]/X[:,2]+cx-img[:,0],f*X[:,1]/X[:,2]+cy-img[:,1]]
        from scipy.optimize import least_squares
        sol=least_squares(res,Rot.from_matrix(Rm).as_rotvec(),loss='huber',f_scale=2.0)
        Rm,t3=pose_of(sol.x)
        chainR.append(Rm); chainT.append(t3); prev=cur
    # loop closure: chained pose at b vs tracked pose at b
    Rb,tb=outR[b],outT[b]; Rerr=Rb@chainR[-1].T; terr=tb-chainT[-1]
    ang=np.degrees(np.linalg.norm(Rot.from_matrix(Rerr).as_rotvec()))
    n=b-a
    for k in range(n):
        w=(k+1)/(n+1); Rc=Rot.from_rotvec(Rot.from_matrix(Rerr).as_rotvec()*w).as_matrix()
        outR[a+k]=Rc@chainR[k]; outT[a+k]=chainT[k]+terr*w; src[a+k]='odometry'
    if n>3:
        yaws=[np.degrees(np.arctan2((chainR[k]@outR[a-1].T)[0,2],(chainR[k]@outR[a-1].T)[2,2])) for k in range(n)]
        print('gap %.1f-%.1fs (%d frames): turned %+.0f° at most, loop-closure error %.1f°%s'%(a/30,b/30,n,yaws[int(np.argmax(np.abs(yaws)))],ang,' (lost track)' if lost else ''))
frs=np.array(sorted(outR)); np.savez('poses_all.npz',R=np.array([outR[k] for k in frs]),tr=np.array([outT[k] for k in frs]),frame=frs,src=np.array([src[k] for k in frs]))
print('posed',len(frs))
