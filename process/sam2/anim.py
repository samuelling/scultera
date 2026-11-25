"""Expression transfer: every sculpt vertex gets weights over the 468 tracked points (through the radial face map),
so each video frame's tracked deltas move the sculpt. Then PCA -> morph targets + per-frame curves (face, head, gaze)."""
import numpy as np, json, sys
from scipy import sparse as sp
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as Rot
fin=np.load('finished.npz'); V=fin['v'].astype(float); F=fin['f']; mat=fin['mat']
fm=np.load('rmap_face.npz'); Tid=fm['Tid']; B=fm['B']; Fs=fm['Fs']; Ws=fm['Ws']; C=fm['C']; RES=float(fm['RES']); AZ0,AZ1,EL0,EL1=fm['ext']
Rhead=np.load('rmap_head.npy'); fc=np.load('face.npz'); D=fc['D']; S=fc['S']
nel,naz=Tid.shape
inside=Tid>=0
dout,(ii,jj)=ndi.distance_transform_edt(~inside,return_indices=True); dout*=RES
q=V-C; rr=np.linalg.norm(q,axis=1); az=np.degrees(np.arctan2(q[:,0],q[:,2])); el=np.degrees(np.arcsin(np.clip(q[:,1]/rr,-1,1)))
gi=np.clip(np.round((el-EL0)/RES-0.5).astype(int),0,nel-1); gj=np.clip(np.round((az-AZ0)/RES-0.5).astype(int),0,naz-1)
ok=((el>EL0)&(el<EL1)&(az>AZ0)&(az<AZ1))
si=ii[gi,gj]; sj=jj[gi,gj]                         # nearest face cell
tri=Tid[si,sj]; bc=B[si,sj]
rsurf=Rhead[gi,gj]
decay=np.exp(-(dout[gi,gj]/np.where(el<-25,11.0,7.0))**2)
depth=np.abs(rr-rsurf)
radial=np.clip(1-(depth-0.6)/0.9,0,1)               # surface layer only (not ears, neck interior)
pocket=(mat==3)|(mat==4)
radial[pocket]=1.0
wgt=np.where(ok&(tri>=0),decay*radial,0.0)
wgt[mat==2]=0                                       # shirt never moves with the face
Wsp=sp.csr_matrix(Ws)
rows=[]; 
idx=np.where(wgt>1e-3)[0]
corner=Fs[tri[idx]]                                 # (n,3) subdivided-vertex ids
Wv=sp.csr_matrix((len(V),468))
for k in range(3):
    Wv=Wv+sp.diags(np.zeros(len(V)))@Wv  # keep shape
M=sp.csr_matrix((wgt[idx]*bc[idx,0],(idx,np.zeros(len(idx),int))),shape=(len(V),1))
parts=[]
for k in range(3):
    Sel=sp.csr_matrix((wgt[idx]*bc[idx,k],(idx,corner[:,k])),shape=(len(V),Ws.shape[0]))
    parts.append(Sel@Wsp)
Wv=(parts[0]+parts[1]+parts[2]).tocsr(); Wv.eliminate_zeros()
print('transfer matrix',Wv.shape,'nnz',Wv.nnz,'moving verts',(np.asarray(Wv.sum(1)).ravel()>0.01).sum())
sp.save_npz('Wv.npz',Wv)
# ---- per-frame curves
T=len(D); Dv=D[:, :468]
X=Dv.reshape(T,-1); mu=X.mean(0)
U,sv,Vt=np.linalg.svd(X-mu,full_matrices=False)
ev=np.cumsum(sv**2)/np.sum(sv**2); K=int(np.searchsorted(ev,0.97)+1); K=min(max(K,12),24)
print('PCA: %d components explain %.1f%% of the face motion'%(K,100*ev[K-1]))
coef=(X-mu)@Vt[:K].T
comps=np.r_[mu[None],Vt[:K]].reshape(K+1,468,3)     # comp 0 = mean offset (applied with weight 1)
# head pose relative to the clip's mean pose, as rotation of the head in the sculpt frame
d=np.load('shape_s.npz'); R=d['R']
Rs=np.array([fc['Rh']@Rr for Rr in R])  # not used directly
Rrel=[]
Rm=Rot.from_matrix(R).mean().as_matrix()
for t in range(T):
    Rd=Rm.T@R[t]                                   # head-frame rotation relative to mean (old frame)
    Rd=fc['Rh'].T@Rd@fc['Rh'] if False else Rd
    Rrel.append(Rot.from_matrix(Rd).as_quat())
Rrel=np.array(Rrel)
eu=Rot.from_quat(Rrel).as_euler('xyz',degrees=True); print('head motion (deg) pitch %.0f..%.0f yaw %.0f..%.0f roll %.0f..%.0f'%(eu[:,0].min(),eu[:,0].max(),eu[:,1].min(),eu[:,1].max(),eu[:,2].min(),eu[:,2].max()))
# gaze per eye: iris centre relative to the globe centre
gaze=[]
for t in range(T):
    g=[]
    for iris,ck in ((468,'eyeR_c'),(473,'eyeL_c')):
        p=S[iris]+D[t,iris]; c=fin[ck]; v=p-c; v/=np.linalg.norm(v)
        g.append([np.degrees(np.arctan2(v[0],v[2])),np.degrees(np.arcsin(v[1]))])
    gaze.append(g)
gaze=np.array(gaze); gaze-=np.median(gaze,0)        # rest = straight ahead
gaze=np.clip(gaze*1.0,-30,30)
print('gaze yaw range %.0f..%.0f pitch %.0f..%.0f'%(gaze[...,0].min(),gaze[...,0].max(),gaze[...,1].min(),gaze[...,1].max()))
jaw=D[:,[152,17,18]].mean(1)*np.array([1,1,1])       # lower teeth ride the chin
np.savez('anim.npz',coef=coef.astype(np.float32),comps=comps.astype(np.float32),quat=Rrel.astype(np.float32),gaze=gaze.astype(np.float32),jaw=jaw.astype(np.float32),K=K,ev=ev[:K])
