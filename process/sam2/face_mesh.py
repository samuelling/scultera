"""Symmetry plane, partial symmetrize, MediaPipe triangle mesh, hole fills, Loop-ish subdivision."""
import numpy as np
d=np.load('shape_s.npz'); S=d['S'].copy(); D=d['D'].copy()
F=np.array([[int(t.split('/')[0])-1 for t in l.split()[1:]] for l in open('models/canonical_face_model.obj') if l.startswith('f ')])
Vc=np.array([[float(x) for x in l.split()[1:4]] for l in open('models/canonical_face_model.obj') if l.startswith('v ')])
# mirror map from the canonical model
mir=np.array([np.argmin(np.linalg.norm(Vc-Vc[i]*[-1,1,1],axis=1)) for i in range(468)])
mir=np.r_[mir,[473,474,475,476,477,468,469,470,471,472]]   # irises swap (centre first)
print('mirror map self-consistent:',(mir[mir]==np.arange(478)).mean())
# fit the symmetry plane: rigid transform that best matches S to its own mirror image
def rigid(A,B):
    ma=A.mean(0);mb=B.mean(0);U,s,Vt=np.linalg.svd((A-ma).T@(B-mb));Dm=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]);R=(U@Dm@Vt).T;return R,mb-R@ma
X=S[:468]; Mx=(X*[-1,1,1])[mir[:468]]
R,t=rigid(Mx,X)          # Mx mapped onto X; the half-rotation removes yaw/roll bias
ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
from scipy.spatial.transform import Rotation as Rot
Rh=Rot.from_rotvec(Rot.from_matrix(R).as_rotvec()/2).as_matrix()
m0=S[:468].mean(0).copy(); S=(S-m0)@Rh.T+m0
sh1=np.median((S[:468,0]+S[mir[:468],0])/2); S[:,0]-=sh1
D=D@Rh.T
asym=np.linalg.norm(S-(S*[-1,1,1])[mir],axis=1)
print('symmetry plane correction %.1f deg; residual asymmetry mean %.2f cm, max %.2f (lm %d)'%(ang/2,asym.mean()/2,asym.max()/2,asym.argmax()))
KEEP=0.25                                    # keep a quarter of his own asymmetry
Ssym=0.5*(S+(S*[-1,1,1])[mir]); S=Ssym+KEEP*(S-Ssym)
# re-centre: origin between irises
sh2=((S[468]+S[473])/2)[0]; S[:,0]-=sh2
# eyes at rest: his typical (median) lid opening across the clip, not the weighted mean
EYE_RINGS=[[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]]
med=np.median(S[None]+D,0)
S0e=S.copy()
for ring in EYE_RINGS:
    S[ring,1]=med[ring,1]; S[ring,2]=0.5*(S[ring,2]+med[ring,2])
D=D+(S0e-S)[None]
print('rest eye aperture now %.2f / %.2f cm'%(S[159,1]-S[145,1],S[386,1]-S[374,1]))
UP=[191,80,81,82,13,312,311,310,415]; LO=[95,88,178,87,14,317,402,318,324]
S0lip=S.copy()
for u,l in zip(UP,LO):
    gap=S[u,1]-S[l,1]; mid=(S[u]+S[l])/2
    S[u]=S[u]+(mid-S[u])*0.6; S[l]=S[l]+(mid-S[l])*0.6
D=D+(S0lip-S)[None]           # expressions keep their absolute lip positions
print('lips sealed at rest: gap at centre now %.2f cm'%(S[13,1]-S[14,1]))
np.savez('face.npz',S=S,D=D,F=F,mir=mir,Rh=Rh,m0=m0,shx=sh1+sh2)
print('cheek x', S[234,0].round(2),S[454,0].round(2))
