import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
D=json.load(open('lm_all.json'))
names=sorted(D)
def to3(L):
    L=np.array(L);c=(L[468]+L[473])/2;s=6.3/np.linalg.norm(L[468]-L[473])
    return np.stack([(L[:,0]-c[0])*s,(L[:,2]-c[2])*s,-(L[:,1]-c[1])*s],1)
S={n:to3(D[n]['lm']) for n in names}
STABLE=[10,151,9,8,168,6,197,195,5,4,1,33,133,362,263,234,454,127,356,93,323,21,251,54,284,103,332,67,297,109,338]
def procrustes(A,B):  # find s,R,t: s*R@A+t ~ B  (on stable pts)
    a=A[STABLE];b=B[STABLE];ma=a.mean(0);mb=b.mean(0);a0=a-ma;b0=b-mb
    U,Sg,Vt=np.linalg.svd(a0.T@b0);d=np.sign(np.linalg.det(U@Vt));Dm=np.diag([1,1,d])
    R=(U@Dm@Vt).T;s=(Sg*np.diag(Dm)).sum()/(a0**2).sum();t=mb-s*R@ma
    return s,R,t
ref=S['p01_front_smile'].copy()
for it in range(6):
    A={};P={}
    for n in names:
        s,R,t=procrustes(S[n],ref);A[n]=(s*(R@S[n].T)).T+t;P[n]=R
    ref=np.median(np.stack([A[n] for n in names]),0)
# region weights on the reference shape
mouthC=(ref[13]+ref[14])/2
def wreg(c,r): return np.exp(-(np.linalg.norm(ref-c,axis=1)/r)**2)
expr_mask=np.clip(wreg(mouthC,3.2)+wreg(ref[159],2.2)+wreg(ref[386],2.2)+wreg(ref[105],2.0)+wreg(ref[334],2.0)+wreg(ref[152],2.6)+wreg(ref[50],2.2)+wreg(ref[280],2.2),0,1)
neutral=0.7*A['p02_neutral_lookdown']+0.3*A['p06_squint_3q_bw']
med=np.median(np.stack([A[n] for n in names]),0)
ident=med*(1-expr_mask[:,None])+neutral*expr_mask[:,None]
# landmark graph smoothing for deltas
from mediapipe.tasks.python.vision import FaceLandmarksConnections as FC
E=np.array([(c.start,c.end) for c in FC.FACE_LANDMARKS_TESSELATION])
import scipy.sparse as sp
Adj=sp.coo_matrix((np.ones(len(E)*2),(np.r_[E[:,0],E[:,1]],np.r_[E[:,1],E[:,0]])),shape=(478,478)).tocsr()
Adj=sp.diags(1/np.maximum(Adj.sum(1).A1,1))@Adj
def smooth(X,it=2,lam=0.5):
    for _ in range(it): X=X+lam*(Adj@X-X)
    return X
delta={n:smooth(A[n]-ident)*expr_mask[:,None] for n in names}
pose={}
for n in names:
    e=Rot.from_matrix(P[n].T).as_euler('xyz',degrees=True)   # photo head rotation relative to ref
    pose[n]=dict(pitch=float(e[0]),roll=float(e[1]),yaw=float(e[2]))
    print(n,'pose(pitch,roll,yaw)',np.round(e,1),'max delta %.2f cm'%np.abs(delta[n]).max())
np.savez('aligned.npz',ident=ident,med=med,names=names,**{'d_'+n:delta[n] for n in names},**{'A_'+n:A[n] for n in names},edges=E,expr_mask=expr_mask)
json.dump(pose,open('poses.json','w'),indent=1)
# per-landmark spread across photos in stable region = consistency check
spread=np.std(np.stack([A[n] for n in names]),0).mean(1)
print('stable-pt spread (cm): %.3f   whole-face mean spread %.3f'%(spread[STABLE].mean(),spread.mean()))
print('ident nose tip',ident[1].round(2),'chin',ident[152].round(2),'cheeks',ident[234].round(2),ident[454].round(2))
