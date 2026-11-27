"""Take 4, step 7: the whole 47-second clip as motion. Face: tracked points drive every vertex (PCA-compressed).
Head: each frame's rotation split into the chair's turn (about the chair axis) and what the neck did."""
import numpy as np, sys, json
sys.path.insert(0,'proc'); from common import *
import sdf4
from scipy import sparse as sp, ndimage as ndi
from scipy.spatial.transform import Rotation as Rot
fin=np.load('finished4.npz'); V=fin['v'].astype(float); mat=fin['mat']; fm=sdf4.fm; Tid=fm['Tid']; Bc=fm['B']; Fs=fm['Fs']; Ws=fm['Ws']; S=fc['S']
ins=Tid>=0; dout,(ii,jj)=ndi.distance_transform_edt(~ins,return_indices=True); dout*=sdf4.RES; nel,naz=Tid.shape
rr,az,el=sdf4.angles(V); gi=np.clip(np.round((el-sdf4.EL0)/sdf4.RES-0.5).astype(int),0,nel-1); gj=np.clip(np.round((az-sdf4.AZ0)/sdf4.RES-0.5).astype(int),0,naz-1)
si,sj=ii[gi,gj],jj[gi,gj]; tri=Tid[si,sj]; bc=Bc[si,sj]; wgt=np.exp(-(dout[gi,gj]/np.where(el<-25,11.0,7.0))**2); wgt[mat==2]=0; wgt[tri<0]=0
idxv=np.where(wgt>1e-3)[0]; Wsp=sp.csr_matrix(Ws); corner=Fs[tri[idxv]]
Wv=sum(sp.csr_matrix((wgt[idxv]*bc[idxv,k],(idxv,corner[:,k])),shape=(len(V),Ws.shape[0]))@Wsp for k in range(3)).tocsr(); sp.save_npz('Wv4.npz',Wv)
good=GOOD; idx=sh['idx']; Dx=fc['D'][:,:468]; kg=np.where(good)[0]; tg=idx[kg]; N=1421
X=Dx[kg].reshape(len(kg),-1); mu=np.median(X,0)*0     # rest face is the reference: no mean offset
U,sv,Vt=np.linalg.svd(X,full_matrices=False); ev=np.cumsum(sv**2)/np.sum(sv**2); K=16; cg=X@Vt[:K].T
print('moving verts %d; PCA: %d shapes keep %.1f%% of the tracked face motion'%((np.asarray(Wv.sum(1)).ravel()>0.01).sum(),K,100*ev[K-1]))
T=np.arange(N); dist=np.abs(T[:,None]-tg[None,:]).min(1); fade=np.ones(N); trk=dist==0; i=0
while i<N:      # the face eases to rest only across long gaps (he faces away); short ones (mid-yawn) are bridged
    if trk[i]: i+=1; continue
    j=i
    while j<N and not trk[j]: j+=1
    if j-i>45: x=np.clip(np.minimum(np.arange(i,j)-i+1,j-np.arange(i,j))/12,0,1); fade[i:j]=1-x*x*(3-2*x)
    i=j
coef=np.stack([np.interp(T,tg,cg[:,k]) for k in range(K)],1)*fade[:,None]; coef=ndi.gaussian_filter1d(coef,0.8,axis=0)
# head: chair turn (twist about the chair axis) + neck (the rest), relative to the frontal key frame
p=np.load('poses4.npz'); R,n=p['R'],p['n']; tf=json.load(open('work/texture.json'))['front']; Rref=R[tf]; nh=Rref.T@n
if nh[1]<0: nh=-nh
psi=[];swing=[]
for t in range(N):
    Rd=Rref.T@R[t]; q=Rot.from_matrix(Rd).as_quat(); psi.append(2*np.arctan2(q[:3]@nh,q[3]))
psi=np.unwrap(np.array(psi)); psi-=psi[tf]
for t in range(N): swing.append(Rot.from_matrix(Rot.from_rotvec(-nh*psi[t]).as_matrix()@(Rref.T@R[t])).as_quat())
swing=np.array(swing); swing[np.sum(swing*swing[tf],1)<0]*=-1; swing=ndi.gaussian_filter1d(swing,1.0,axis=0); swing/=np.linalg.norm(swing,axis=1,keepdims=True)
psi_s=ndi.gaussian_filter1d(np.degrees(psi),1.0); sa=2*np.degrees(np.arccos(np.clip(np.abs(swing[:,3]),0,1)))
print('chair axis in head coords',nh.round(3),' chair turn %.0f..%.0f deg; neck bend up to %.0f deg'%(psi_s.min(),psi_s.max(),sa.max()))
gz=np.zeros((N,2,2))
for e,(iris,ck) in enumerate(((468,'eyeR_c'),(473,'eyeL_c'))):
    c=fin[ck]; v=S[iris]+fc['D'][kg,iris]-c; v/=np.linalg.norm(v,axis=1,keepdims=True); g=np.c_[np.degrees(np.arctan2(v[:,0],v[:,2])),np.degrees(np.arcsin(v[:,1]))]; g-=np.median(g,0)
    for c2 in range(2): gz[:,e,c2]=np.clip(ndi.gaussian_filter1d(np.interp(T,tg,g[:,c2])*fade,1.0),-28,28)
jaw=np.stack([np.interp(T,tg,fc['D'][kg][:,[152,17,18]].mean(1)[:,c]) for c in range(3)],1)*fade[:,None]
sd=coef.std(0); np.savez('anim5.npz',coef=coef,comps=Vt[:K].reshape(K,468,3),sd=sd,psi=psi_s,swing=swing,gaze=gz,jaw=jaw,K=K,nh=nh,ev=ev[:K],tracked=(dist==0))
json.dump(dict(fps=30,frames=N,K=K,coef=np.round(coef/sd,2).tolist(),yaw=np.round(psi_s,1).tolist(),swing=np.round(swing,4).tolist(),gaze=np.round(gz,1).tolist(),jaw=np.round(jaw*0.01,5).tolist(),
    tracked=(dist==0).astype(int).tolist(),axis=[round(float(x),4) for x in nh],front=int(tf)),open('motion4.json','w'),separators=(',',':'))
