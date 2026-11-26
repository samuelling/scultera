"""Radial map of the visual hull around the head centre; round off the unseen back wedge with a per-ring Fourier fit."""
import numpy as np
from scipy import ndimage as ndi
fm=np.load('rmap_face.npz'); fc=np.load('face.npz'); C=fm['C']; AZ0,AZ1,EL0,EL1=fm['ext']
h=np.load('hull_t2.npz'); occ=h['occ'].copy(); xs,ys,zs=h['xs'],h['ys'],h['zs']
occ[:,ys<-14.5,:]=False                        # head only; neck/bust are built separately
occ=ndi.gaussian_filter(occ.astype(np.float32),1.2)
RES=0.5; az=np.arange(-180,180,RES)+RES/2; el=np.arange(-89,89,RES)+RES/2
A,E=np.meshgrid(np.radians(az),np.radians(el))
Dm=np.stack([np.cos(E)*np.sin(A),np.sin(E),np.cos(E)*np.cos(A)],-1)
rs=np.arange(1.5,24,0.04).astype(np.float32); Rh=np.full(A.shape,np.nan)
for i0 in range(0,len(el),12):
    P=C+Dm[i0:i0+12,:,None,:]*rs[:,None]
    P=P+np.array([fc['shx'],0,0]); P=(P-fc['m0'])@fc['Rh']+fc['m0']       # into the hull's (pre-symmetry) frame
    idx=np.stack([(P[...,0]-xs[0])/(xs[1]-xs[0]),(P[...,1]-ys[0])/(ys[1]-ys[0]),(P[...,2]-zs[0])/(zs[1]-zs[0])]).astype(np.float32)
    val=ndi.map_coordinates(occ,idx.reshape(3,-1),order=1,cval=0).reshape(P.shape[:-1])
    ins=val>0.5; an=ins.any(-1); last=len(rs)-1-np.argmax(ins[...,::-1],-1)
    blk=rs[last].astype(float); blk[~an]=np.nan; Rh[i0:i0+12]=blk
# back wedge: never seen straight on. Fit an ellipsoid skull to the hull where it is well carved (sides, crown),
# and since the hull can only be too big, take the smaller of the two over the back.
from scipy.optimize import least_squares
sel=np.isfinite(Rh)&(np.abs(A)<np.radians(135))&(E>np.radians(-15))&~((np.abs(A)<np.radians(55))&(E<np.radians(40)))
Pts=C+Dm[sel]*Rh[sel][:,None]
def res(p):
    c=np.array([0,p[0],p[1]]); r=p[2:5]; q=(Pts-c)/r; return (np.linalg.norm(q,axis=1)-1)*r.mean()
sol=least_squares(res,[2,-8.5,8,9,9.5]); p=sol.x; ec=np.array([0,p[0],p[1]]); er=p[2:5]
o=(C-ec)/er; dd=Dm/er; aa=(dd*dd).sum(-1); bb=2*(o*dd).sum(-1); cc=(o*o).sum()-1
Rell=(-bb+np.sqrt(bb*bb-4*aa*cc))/(2*aa)
print('skull ellipsoid fit to the carved sides: width %.1f length %.1f height %.1f cm, rms %.2f cm'%(2*er[0],2*er[2],2*er[1],np.sqrt((sol.fun**2).mean())))
ad=np.abs(np.degrees(A)); ed=np.degrees(E)
wb=np.clip((ad-108)/30,0,1)                                   # back half: the fitted skull
wl=np.clip((-ed-20)/12,0,1)*np.clip((ad-55)/20,0,1)           # under the skull behind the jaw: skull (the neck fills in)
wt=np.clip((ed-45)/20,0,1)*np.clip((ad-25)/30,0,1)          # crown behind the hairline: skull
w=np.maximum(np.maximum(wb,wl),wt); w=w*w*(3-2*w)
Rhs=ndi.gaussian_filter(np.where(np.isfinite(Rh),Rh,Rell),3.0)   # the carve is streaky from small pose errors: smooth ~1.5 deg
Rr=Rhs*(1-w)+Rell*w
Rr=ndi.gaussian_filter(Rr,1.0)
np.save('skull_ell3.npy',np.r_[ec,er])
np.savez('rmap_hull3.npz',R=Rr,az=az,el=el,RES=RES)
print('hull r: front %.1f  back raw %.1f -> rounded %.1f  top %.1f'%(Rr[np.argmin(abs(el)),np.argmin(abs(az))],Rh[np.argmin(abs(el-20)),np.argmin(abs(abs(az)-179))],Rr[np.argmin(abs(el-20)),np.argmin(abs(abs(az)-179))],Rr[-3].mean()))
