"""Take 4 surface as one signed-distance field: the carve itself (not a radial map), the tracked face cutting its hollows,
optional stereo relief, neck + bust, a few light details. Also: outside-in ray casting of that field for the texture."""
import numpy as np, sys, os
sys.path.insert(0,'proc'); from common import *
from scipy import ndimage as ndi
fm=np.load(B3+'rmap_face.npz'); Rf=fm['Rf']; Lab=fm['Lab']; C=fm['C']; RES=float(fm['RES']); AZ0,AZ1,EL0,EL1=fm['ext']; nel,naz=Rf.shape
S=fc['S']; EL3=np.load(B3+'skull_ell3.npy')
h4=np.load('hull4.npz'); occ=h4['occ_t']; xs,ys,zs=h4['xs'],h4['ys'],h4['zs']; hv=float(xs[1]-xs[0])
SMOOTH=float(os.environ.get('SMOOTH','0.3'))
_raw=((ndi.distance_transform_edt(~occ)-ndi.distance_transform_edt(occ))*hv).astype(np.float32)
_sdB=ndi.gaussian_filter(_raw,1.6/hv); _sd=ndi.gaussian_filter(_raw,SMOOTH/hv); _sdS=ndi.gaussian_filter(_raw,0.75/hv)     # scalp: smoothed harder (the carve is streaky there)
def d_hull(P,ws=None,wb=None):
    Q=toold(P); idx=np.stack([(Q[...,0]-xs[0])/hv,(Q[...,1]-ys[0])/hv,(Q[...,2]-zs[0])/hv]).reshape(3,-1)
    a=ndi.map_coordinates(_sd,idx,order=1,mode='nearest').reshape(P.shape[:-1])
    if ws is None: return a
    a=a*(1-ws)+ndi.map_coordinates(_sdS,idx,order=1,mode='nearest').reshape(P.shape[:-1])*ws
    return a*(1-wb)+ndi.map_coordinates(_sdB,idx,order=1,mode='nearest').reshape(P.shape[:-1])*wb
inside=np.isfinite(Rf); rfp=Rf.copy(); rfp[Lab==2]-=1.0; rfp[Lab==3]-=1.0; rfp[Lab==1]-=1.3
_d,(ii,jj)=ndi.distance_transform_edt(~inside,return_indices=True); RFP=rfp[ii,jj].astype(np.float32)
din=ndi.distance_transform_edt(inside)*RES; WF=np.clip(din/6.0,0,1); WF=(WF*WF*(3-2*WF)).astype(np.float32)
RELIEF=np.zeros((nel,naz),np.float32)
if os.environ.get('RELIEF') and os.path.exists(os.environ['RELIEF']): RELIEF=np.load(os.environ['RELIEF']).astype(np.float32)
def angles(P):
    q=P-C; rr=np.linalg.norm(q,axis=-1); az=np.degrees(np.arctan2(q[...,0],q[...,2])); el=np.degrees(np.arcsin(np.clip(q[...,1]/np.maximum(rr,1e-6),-1,1)))
    return rr,az,el
def fmap(M,az,el,order=1):
    gi=np.clip((el-EL0)/RES-0.5,0,nel-1); gj=np.clip((az-AZ0)/RES-0.5,0,naz-1)
    return ndi.map_coordinates(M,[gi.ravel(),gj.ravel()],order=order,mode='nearest').reshape(az.shape)
def smin(a,b,k): h=np.clip(0.5+0.5*(b-a)/k,0,1); return b*(1-h)+a*h-k*h*(1-h)
def smax(a,b,k): return -smin(-a,-b,k)
def capsule(P,a,b,ra,rb):
    pa=P-a; ba=b-a; h=np.clip((pa@ba)/(ba@ba),0,1); return np.linalg.norm(pa-h[...,None]*ba,axis=-1)-(ra+(rb-ra)*h)
def ellip(P,c,r,Rm=np.eye(3)):
    q=(P-c)@Rm; k0=np.linalg.norm(q/r,axis=-1); k1=np.linalg.norm(q/(r*r),axis=-1); return k0*(k0-1)/np.maximum(k1,1e-9)
def chain(P,pts,ra):
    out=np.full(P.shape[:-1],9.0)
    for i in range(len(pts)-1): out=np.minimum(out,capsule(P,pts[i],pts[i+1],ra,ra))
    return out
def rotx(deg):
    t=np.radians(deg);c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
# measured from the carve: neck cross-section, ear positions
I=np.argwhere(occ&~ndi.binary_erosion(occ)); HS=tonew(np.c_[xs[I[:,0]],ys[I[:,1]],zs[I[:,2]]])
sl=HS[np.abs(HS[:,1]+13.5)<0.3]; NECKC=np.array([0.0,sl[:,2].mean()]); NECKR=float(np.sqrt(np.mean(sl[:,0]**2+(sl[:,2]-NECKC[1])**2)))
EAR={}
for sg in (-1,1):
    e=HS[(HS[:,1]>-6)&(HS[:,1]<3)&(HS[:,0]*sg>0)]; tip=np.percentile(e[:,0]*sg,99.5); pts=e[e[:,0]*sg>tip-0.5]; EAR[sg]=np.array([sg*tip,pts[:,1].mean(),pts[:,2].mean()])
OUTL=[61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
HCUT=-17.2
def sdf(P,parts=('head','face','relief','neck','bust','details')):
    rr,az,el=angles(P); sm=lambda x: np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
    if 'raw' in parts: d=d_hull(P)
    else:
        ws=np.maximum(sm((el-14)/14),sm((np.abs(az)-124)/12))
        # +-70 deg of tracked turn leaves a ridge (at most 6 mm) down the back's centre line: rounded off with a wide blur there
        wb=sm((np.abs(az)-138)/22)*sm((el+40)/20); d=d_hull(P,ws,wb)
    if 'face' in parts:
        w=fmap(WF,az,el); df=(rr-fmap(RFP,az,el))*0.92
        if 'relief' in parts: df=df-fmap(RELIEF,az,el)
        d=d+w*(smax(d,df,0.25)-d)
    d=smax(d,-(P[...,1]-HCUT),0.6)
    # below the jaw keep only the neck column: the shoulders are not rigid with the head, so their carve is smeared
    lim=np.hypot(P[...,0],P[...,2]+8.8)-(6.9+0.2*np.clip(-11.5-P[...,1],0,6)); d=d+sm((-11.0-P[...,1])/2.0)*(smax(d,lim,0.8)-d)
    if 'neck' in parts:
        d=smin(d,capsule(P,np.array([0,-12.0,-8.6]),np.array([0,-21.0,-8.2]),5.4,5.8),1.0)
    if 'bust' in parts:
        to=ellip(P,np.array([0,-25.0,-8.6]),np.array([20.0,8.0,10.5]))
        for sg in (-1,1): to=smin(to,capsule(P,np.array([sg*5.5,-15.8,-9.6]),np.array([sg*16.5,-20.5,-9.0]),3.2,3.0),2.0)
        to=smax(to,np.abs(P[...,0])-17.5,1.5); to=smax(to,-(P[...,1]+28.5),0.8); d=smin(d,to,2.2)
    if 'details' in parts:
        for sg,(nos,cr) in ((-1,(79,98)),(1,(309,327))):
            nc=(S[nos]*0.45+S[cr]*0.35+S[2]*0.2)+np.array([0,-0.12,-0.12]); d=smax(d,-ellip(P,nc,np.array([0.36,0.2,0.5]),rotx(-35)),0.1)
            d=smax(d,-ellip(P,EAR[sg]+np.array([-sg*0.15,-0.2,0.35]),np.array([0.45,1.0,0.7])),0.25)       # concha
        d=smin(d,chain(P,[S[i]+np.array([0,0,-0.055]) for i in OUTL],0.06),0.04)
    return d
def dirs(az,el):
    A,E=np.meshgrid(np.radians(az),np.radians(el)); return np.stack([np.cos(E)*np.sin(A),np.sin(E),np.cos(E)*np.cos(A)],-1)
def trace(D,parts=('head','face','relief','neck','details'),r0=27.0,iters=60):
    """outermost surface along each ray from the head centre, found from outside in"""
    r=np.full(D.shape[:-1],r0); act=np.ones(r.shape,bool)
    for it in range(iters):
        if not act.any(): break
        d=sdf(C+r[act][:,None]*D[act],parts); r[act]-=np.clip(d,-0.5,2.5)*0.9; a2=np.abs(d)>0.004; idx=np.where(act); act[idx]=a2&(r[act]>1.0)
    return r
# conform: the tracked face and the carve disagree by up to 9 mm at the face's rim (measured). The carve is the measured
# outline, so the face is shifted to meet it there; the shift fades toward the centre, where the two already agree.
import cv2 as _cv
_bj=np.where(inside.any(0))[0]; _bi=np.where(inside.any(1))[0]; _j0,_j1=_bj[0]-10,_bj[-1]+10; _i0,_i1=_bi[0]-10,_bi[-1]+10
try: _rh=np.load('work/rhull_face.npy')
except FileNotFoundError:
    _rh=trace(dirs(AZ0+RES*(np.arange(_j0,_j1)+0.5),EL0+RES*(np.arange(_i0,_i1)+0.5)),parts=('head',)).astype(np.float32); np.save('work/rhull_face.npy',_rh)
_band=(inside&(din<3)&(Lab==0))[_i0:_i1,_j0:_j1].astype(np.float32); _dl=np.nan_to_num(Rf[_i0:_i1,_j0:_j1]-_rh)*_band
_ext=_cv.GaussianBlur(_dl,(0,0),50)/np.maximum(_cv.GaussianBlur(_band,(0,0),50),1e-4)
CONF=(_ext*np.exp(-(din[_i0:_i1,_j0:_j1]/14.0)**2)).astype(np.float32)
if os.environ.get('NOCONF') is None: RFP[_i0:_i1,_j0:_j1]-=CONF
if __name__=='__main__':
    print('conform shift at the rim: %.1f .. %.1f mm'%(10*_ext[_band>0].min(),10*_ext[_band>0].max()))
    print('neck radius %.1f cm at z %.1f; ears at'%(NECKR,NECKC[1]),EAR[-1].round(1),EAR[1].round(1))
    from skimage import measure; from prev import sheet
    H=0.25; gx=np.arange(-22,22+H,H); gy=np.arange(-31,14+H,H); gz=np.arange(-21,9+H,H); vol=np.empty((len(gx),len(gy),len(gz)),np.float32)
    Y,Z=np.meshgrid(gy,gz,indexing='ij')
    for i,x in enumerate(gx): vol[i]=sdf(np.stack([np.full_like(Y,x),Y,Z],-1))
    v,f,_,_=measure.marching_cubes(vol,0,spacing=(H,H,H)); v+=[gx[0],gy[0],gz[0]]
    sheet([(v,f,(0.78,0.68,0.6))],'work/prev_sdf.png',views=((0,0),(35,-5),(90,0),(150,0),(180,0)),size=330,span=40,center=(0,-7,-7))
