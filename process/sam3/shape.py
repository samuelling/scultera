"""Take 2, step 1: one rigid face shape from 358 video frames.
Per-frame perspective back-projection -> rigid alignment -> neutral-weighted average ->
bundle adjustment on the 2D tracks (shape shared by all frames, pose per frame)."""
import numpy as np, json
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot
d=np.load('track.npz'); LM=d['lm']; M=d['M']; BS=d['bs']; BN=list(d['bsnames']); W,H=int(d['W']),int(d['H'])
OK=d['ok']; idx_ok=np.where(OK)[0]; LM=LM[OK]; M=M[OK]; BS=BS[OK]
T=len(LM); f=(H/2)/np.tan(np.radians(31.5)); cx,cy=W/2,H/2
def back(t):
    u,v,z=LM[t].T; Z0=-M[t][2,3]; Z=Z0+z*Z0/f
    return np.stack([(u-cx)*Z/f,(v-cy)*Z/f,Z],1)          # camera frame, cm (y down, z away)
P=np.array([back(t) for t in range(T)])
STABLE=[10,151,9,8,168,6,197,195,5,4,1,33,133,362,263,234,454,127,356,93,323,21,251,54,284,103,332,67,297,109,338,152,175]
def rigid(A,B,idx):
    a=A[idx];b=B[idx];ma=a.mean(0);mb=b.mean(0)
    U,S,Vt=np.linalg.svd((a-ma).T@(b-mb));D=np.diag([1,1,np.sign(np.linalg.det(U@Vt))])
    R=(U@D@Vt).T; return R, mb-R@ma
# head frame: x right(subject's left on image right), y up, z toward camera, origin between eyes
ref=P[np.argmin(np.abs(BS[:,1:]).sum(1))].copy()
for it in range(5):
    Al=[];Rs=[]
    for t in range(T):
        R,tr=rigid(P[t],ref,STABLE);Al.append(P[t]@R.T+tr);Rs.append(R)
    Al=np.array(Al);ref=np.median(Al,0)
# canonical orientation: eyes level, face looking +z
c=(ref[468]+ref[473])/2; xa=ref[473]-ref[468]; xa/=np.linalg.norm(xa)   # 468=subject right iris (image left)
up=ref[10]-ref[152]; up-=xa*(up@xa); up/=np.linalg.norm(up); fw=np.cross(xa,up)
Rc=np.stack([xa,up,fw])           # rows: new axes in camera coords
if (ref[1]-c)@fw<0: Rc[2]*=-1; Rc[0]*=-1
Al=(Al-c)@Rc.T; ref=(ref-c)@Rc.T
# per-frame expression load by region -> neutral weights
def b(n): return BS[:,BN.index(n)]
mouth=sum(b(n) for n in BN if n.startswith(('mouth','jaw','cheek')))
eyes=sum(b(n) for n in ['eyeBlinkLeft','eyeBlinkRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight'])
brow=sum(b(n) for n in BN if n.startswith('brow'))
y=ref[:,1]; mouth_reg=np.clip((ref[1,1]-y)/3.0,0,1)          # below nose tip -> mouth-driven
eye_reg=np.exp(-np.minimum(np.linalg.norm(ref-ref[159],axis=1),np.linalg.norm(ref-ref[386],axis=1))**2/2.0)
brow_reg=np.exp(-np.minimum(np.linalg.norm(ref-ref[105],axis=1),np.linalg.norm(ref-ref[334],axis=1))**2/2.5)
load=mouth[:,None]*mouth_reg[None]+eyes[:,None]*eye_reg[None]+brow[:,None]*brow_reg[None]
w=np.exp(-4*(load-load.min(0)))                                   # T x 478
neutral0=(w[...,None]*Al).sum(0)/w.sum(0)[:,None]
print('neutral frames (mouth) best t =',np.argsort(mouth)[:5]/30)
# ---- bundle adjustment on the 2D tracks
uv=LM[:,:,:2]; sub=np.arange(0,T,4); n=len(sub); N=478
Rinv=Rc.T  # head->camera rotation part
def cam_pose(t):
    R,tr=rigid(neutral0,P[t],STABLE); return Rot.from_matrix(R).as_rotvec(),tr
x0=[];
for t in sub: rv,tr=cam_pose(t); x0+= [*rv,*tr]
x0=np.r_[x0,neutral0.ravel()]
wt=np.sqrt(w[sub])                                               # down-weight expressive landmarks
zprior=0.15                                                       # px-equivalent weight on MediaPipe's own depth
def resid(x):
    S=x[6*n:].reshape(N,3); out=[]
    for k,t in enumerate(sub):
        R=Rot.from_rotvec(x[6*k:6*k+3]).as_matrix(); X=S@R.T+x[6*k+3:6*k+6]
        pu=f*X[:,0]/X[:,2]+cx; pv=f*X[:,1]/X[:,2]+cy
        out.append(((pu-uv[t,:,0])*wt[k])); out.append(((pv-uv[t,:,1])*wt[k]))
    out.append(zprior*10*(S[:,2]-neutral0[:,2]))                 # keep depth near the learned prior
    out.append(30*(S[STABLE[:3]]-neutral0[STABLE[:3]]).ravel())  # pin gauge (pos/scale)
    return np.concatenate(out)
J=lil_matrix((2*n*N+N+9,len(x0)),dtype=int)
for k in range(n):
    for r in (0,1):
        rows=np.arange((2*k+r)*N,(2*k+r+1)*N); J[rows[:,None],np.arange(6*k,6*k+6)[None]]=1
        J[rows,6*n+3*np.arange(N)]=1;J[rows,6*n+3*np.arange(N)+1]=1;J[rows,6*n+3*np.arange(N)+2]=1
J[2*n*N+np.arange(N),6*n+3*np.arange(N)+2]=1
for i,j in enumerate(STABLE[:3]):
    for a in range(3): J[2*n*N+N+3*i+a,6*n+3*j+a]=1
r0=resid(x0); print('BA start rms px %.2f'%np.sqrt((r0[:2*n*N]**2).mean()))
sol=least_squares(resid,x0,jac_sparsity=J.tocsr(),method='trf',x_scale='jac',max_nfev=40,verbose=0)
r1=sol.fun; print('BA end   rms px %.2f'%np.sqrt((r1[:2*n*N]**2).mean()))
S=sol.x[6*n:].reshape(N,3)
dz=S[:,2]-neutral0[:,2]; print('depth correction: mean |dz| %.2f mm, max %.1f mm (lm %d)'%(10*np.abs(dz).mean(),10*np.abs(dz).max(),np.abs(dz).argmax()))
# poses for every frame against the BA shape (camera-from-head)
poses=[]
for t in range(T):
    R,tr=rigid(S,P[t],STABLE); poses.append((R,tr))
# expression deltas per frame in head frame
D=np.array([(P[t]-poses[t][1])@poses[t][0] - S for t in range(T)])
np.savez('shape.npz',idx_ok=idx_ok,S=S,neutral0=neutral0,ref=ref,D=D,R=np.array([p[0] for p in poses]),tr=np.array([p[1] for p in poses]),f=f,cx=cx,cy=cy,w=w,Rc=Rc)
print('ipd %.2f  face h (10->152) %.2f  face w (234-454) %.2f  nose depth (1 vs 168) %.2f cm'%(
    np.linalg.norm(S[468]-S[473]),np.linalg.norm(S[10]-S[152]),np.linalg.norm(S[234]-S[454]),S[1,2]-S[168,2]))
