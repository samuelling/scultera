"""One-number depth test: scale the averaged face's depth by k and see which k the video's head turns agree with."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
d=np.load('shape.npz'); N0=d['neutral0']; f=float(d['f']); cx=float(d['cx']); cy=float(d['cy']); w=d['w']
tk=np.load('track.npz'); LM=tk['lm'][tk['ok']]; T=len(LM)
def proj(x,P):
    X=P@Rot.from_rotvec(x[:3]).as_matrix().T+x[3:]; return np.stack([f*X[:,0]/X[:,2]+cx,f*X[:,1]/X[:,2]+cy],1)
res={}
for fs in [0.8,1.0,1.25,1.6]:
  ff=f*fs
  for k in [0.85,1.0,1.15]:
    S=N0.copy(); S[:,2]=S[:,2]*k
    x=np.r_[np.pi,0,0,0,0,45.0]; E=[]
    for t in range(0,T,6):
        wt=np.sqrt(w[t])[:,None]
        def r(x):
            X=S@Rot.from_rotvec(x[:3]).as_matrix().T+x[3:]
            return ((np.stack([ff*X[:,0]/X[:,2]+cx,ff*X[:,1]/X[:,2]+cy],1)-LM[t,:,:2])*wt).ravel()
        s=least_squares(r,x,method='lm');x=s.x; E.append(np.sqrt((s.fun**2).mean()))
    res[(fs,k)]=np.mean(E); print('focal x%.2f depth x%.2f  weighted rms px %.3f'%(fs,k,np.mean(E)),flush=True)
