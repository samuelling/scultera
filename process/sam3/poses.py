"""PnP per frame against the shared shape (reprojection, not 3D fit), then expression deltas."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
d=np.load('shape.npz'); S0=d['neutral0'].copy(); S0[:,2]*=1.10; f=float(d['f']); cx=float(d['cx']); cy=float(d['cy']); w=d['w']
print('neutral0 ipd %.2f   BA ipd %.2f'%(np.linalg.norm(d['neutral0'][468]-d['neutral0'][473]),np.linalg.norm(S0[468]-S0[473])))
sc=6.3/np.linalg.norm(S0[468]-S0[473]); S=S0*sc
tk=np.load('track.npz'); OK=tk['ok']; LM=tk['lm'][OK]; M=tk['M'][OK]; T=len(LM)
STABLE=[10,151,9,8,168,6,197,195,5,4,1,33,133,362,263,234,454,127,356,93,323,21,251,54,284,103,332,67,297,109,338,152,175]
def proj(x,P):
    X=P@Rot.from_rotvec(x[:3]).as_matrix().T+x[3:]; return np.stack([f*X[:,0]/X[:,2]+cx,f*X[:,1]/X[:,2]+cy],1)
x=np.r_[np.pi,0,0,0,0,45.0]   # face toward camera: rotate 180 about x (y up->down, z toward cam -> away)
Rs=[];ts=[];err=[]
def rigid3(A,B):
    ma=A.mean(0);mb=B.mean(0);U,Sg,Vt=np.linalg.svd((A-ma).T@(B-mb));Dm=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]);R=(U@Dm@Vt).T;return R,mb-R@ma
f0=f
for t in range(T):
    wt=np.sqrt(w[t])[:,None]
    u,v,z=LM[t].T; Z0=-M[t][2,3]; Zz=Z0+z*Z0/f0; Pc=np.stack([(u-cx)*Zz/f0,(v-cy)*Zz/f0,Zz],1)
    # MediaPipe's metric depth is in its own scale: scale Pc to our face size first
    sc_=np.linalg.norm(S[468]-S[473])/max(np.linalg.norm(Pc[468]-Pc[473]),1e-6); Pc=Pc*sc_
    Ri,ti=rigid3(S,Pc); x0=np.r_[Rot.from_matrix(Ri).as_rotvec(),ti]
    r=lambda x: ((proj(x,S)-LM[t,:,:2])*wt).ravel()
    best=None
    for xi in (x0,x):
        s=least_squares(r,xi,method='lm')
        if best is None or s.cost<best.cost: best=s
    x=best.x
    e=np.linalg.norm(proj(x,S)-LM[t,:,:2],axis=1); err.append(e.mean())
    Rs.append(Rot.from_rotvec(x[:3]).as_matrix()); ts.append(x[3:].copy())
Rs=np.array(Rs); ts=np.array(ts); err=np.array(err)
print('PnP reprojection px: mean %.2f  95pct %.2f'%(err.mean(),np.percentile(err,95)),' depth %.1f-%.1f cm'%(ts[:,2].min(),ts[:,2].max()))
# expression deltas: back-project each landmark at its tracked (u,v) with MediaPipe relative depth around the PnP depth
D=[]
for t in range(T):
    X=S@Rs[t].T+ts[t]; Zc=X[:,2]
    u,v,z=LM[t].T; zrel=z-z[STABLE].mean(); Zm=Zc[STABLE].mean()
    Z=Zm+zrel*Zm/f
    P=np.stack([(u-cx)*Z/f,(v-cy)*Z/f,Z],1)
    Ph=(P-ts[t])@Rs[t]                       # into head frame with the PnP pose
    # remove residual similarity on stable points (MediaPipe depth scale != ours)
    a=Ph[STABLE];b=S[STABLE];ma=a.mean(0);mb=b.mean(0)
    U,Sg,Vt=np.linalg.svd((a-ma).T@(b-mb));Dm=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]);Rr=(U@Dm@Vt).T
    s_=(Sg*np.diag(Dm)).sum()/((a-ma)**2).sum()
    D.append(s_*(Ph-ma)@Rr.T+mb-S)
D=np.array(D)
e=np.array([Rot.from_matrix(Rs[t]@Rs[0].T).as_euler('xyz',degrees=True) for t in range(T)])
print('head motion rel. to frame 0 (deg): pitch %.0f..%.0f yaw %.0f..%.0f roll %.0f..%.0f'%(e[:,0].min(),e[:,0].max(),e[:,1].min(),e[:,1].max(),e[:,2].min(),e[:,2].max()))
print('max expression delta %.2f cm (lm %d)'%(np.abs(D).max(),np.abs(D).max(2).max(0).argmax()))
good=(err<6)&(ts[:,2]>40)&(ts[:,2]<250)&(np.abs(D).max((1,2))<3.0)
print('good frames %d / %d'%(good.sum(),len(good)),' depth median %.0f cm'%np.median(ts[good,2]))
np.savez('shape_s.npz',good=good,S=S,R=Rs,tr=ts,D=D,f=f,cx=cx,cy=cy,w=w,sc=sc,err=err,idx=np.where(OK)[0])
