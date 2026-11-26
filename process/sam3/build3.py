"""Take 2 blockout + surface: one star-shaped head from radial maps, smooth-unioned with jaw, neck, bust, ears."""
import numpy as np, sys, json, time, os
sys.path.insert(0,'proc')
from scipy import ndimage as ndi
from skimage import measure
H=float(sys.argv[1]) if len(sys.argv)>1 else 0.2
STAGE=sys.argv[2] if len(sys.argv)>2 else 'final'
fm=np.load('rmap_face.npz'); Rf=fm['Rf'].copy(); Lab=fm['Lab']; C=fm['C']; RES=float(fm['RES']); AZ0,AZ1,EL0,EL1=fm['ext']
S=np.load('face.npz')['S']
nel,naz=Rf.shape
az=AZ0+RES*(np.arange(naz)+0.5); el=EL0+RES*(np.arange(nel)+0.5); A,E=np.meshgrid(np.radians(az),np.radians(el))
Dm=np.stack([np.cos(E)*np.sin(A),np.sin(E),np.cos(E)*np.cos(A)],-1)
LOG=[]
def log(phase,tool,target,note,**p): LOG.append(dict(phase=phase,tool=tool,target=target,note=note,params=p))
# ---- head from the visual hull (rounded back), face hollows from the tracked surface
hm=np.load('rmap_hull3.npz'); HR=hm['R']; haz=hm['az']; hel=hm['el']; HRES=float(hm['RES'])
def r_hull_dir(D):
    a=np.degrees(np.arctan2(D[...,0],D[...,2])); e=np.degrees(np.arcsin(np.clip(D[...,1],-1,1)))
    gi=np.clip((e-hel[0])/HRES,0,len(hel)-1); gj=((a-haz[0])/HRES)%len(haz)
    return ndi.map_coordinates(np.c_[HR,HR[:,:1]],[gi,gj],order=1,mode='nearest')
Rc=r_hull_dir(Dm)
log('blockout','Carve','visual hull','the head carved from its outline in 745 posed frames, turning from -83° to +71°',frames=745)
log('blockout','Round','back of head','the camera never saw the back straight on: each ring refit there from the sides',sector_deg=64)
rf=Rf.copy(); inside=np.isfinite(rf)
rf[Lab==2]-=1.0; rf[Lab==3]-=1.0; rf[Lab==1]-=1.3
din=ndi.distance_transform_edt(inside)*RES
w=np.clip(din/4.0,0,1); w=w*w*(3-2*w)
rfc=np.where(inside,np.nan_to_num(rf),Rc)
k=0.25; hh=np.clip(0.5+0.5*(Rc-rfc)/k,0,1); face_in=Rc*(1-hh)+rfc*hh-k*hh*(1-hh)     # smooth min(face, hull)
face_in=np.where((Lab>0)&inside,np.minimum(rfc,Rc),face_in)                            # pockets stay crisp
R=np.where(inside,w*face_in+(1-w)*Rc,Rc)
if STAGE=='hull': R=Rc.copy()
log('surface','Fit','face','tracked face surface pressed into the hull: the hull bounds the outside, the tracked points carve the hollows (eyes, lips, nose wings)',verts=int(len(fm['Vs'])))
np.save('rmap_head.npy',R)
# ---- volume
def r_lookup(P):
    q=P-C; rr=np.linalg.norm(q,axis=-1); a=np.degrees(np.arctan2(q[...,0],q[...,2])); e=np.degrees(np.arcsin(np.clip(q[...,1]/np.maximum(rr,1e-6),-1,1)))
    gi=(e-EL0)/RES-0.5; gj=(a-AZ0)/RES-0.5
    ok=(gi>=0)&(gi<=nel-1)&(gj>=0)&(gj<=naz-1)
    out=r_hull_dir(q/np.maximum(rr,1e-6)[...,None])
    out[ok]=ndi.map_coordinates(R,[gi[ok],gj[ok]],order=1)
    return rr,out
def capsule(P,a,b,ra,rb):
    pa=P-a; ba=b-a; h=np.clip((pa@ba)/(ba@ba),0,1); return np.linalg.norm(pa-h[...,None]*ba,axis=-1)-(ra+(rb-ra)*h)
def ellip(P,c,r,Rm=np.eye(3)):
    q=(P-c)@Rm; k0=np.linalg.norm(q/r,axis=-1); k1=np.linalg.norm(q/(r*r),axis=-1); return k0*(k0-1)/np.maximum(k1,1e-9)
def sminv(a,b,k): h=np.clip(0.5+0.5*(b-a)/k,0,1); return b*(1-h)+a*h-k*h*(1-h)
def smaxv(a,b,k): return -sminv(-a,-b,k)
def rot(ax,deg):
    t=np.radians(deg);c,s=np.cos(t),np.sin(t)
    return {'x':np.array([[1,0,0],[0,c,-s],[0,s,c]]),'y':np.array([[c,0,s],[0,1,0],[-s,0,c]]),'z':np.array([[c,-s,0],[s,c,0],[0,0,1]])}[ax]
gon=lambda sg: np.array([sg*5.7,-7.6,-7.2])
chin=S[152]+np.array([0,1.3,-1.5])

def chain(P,pts,ra,rb=None):
    rb=ra if rb is None else rb; out=np.full(P.shape[:-1],9.0); n=len(pts)-1
    for i in range(n): out=np.minimum(out,capsule(P,pts[i],pts[i+1],ra+(rb-ra)*i/n,ra+(rb-ra)*(i+1)/n))
    return out
def off(i,dz=0,dy=0,dx=0): return S[i]+np.array([dx,dy,dz])
DET=[]
def detail(name,tool,note,**p): DET.append(name); log('detail',tool,name,note,**p)
detail('nostrils','Carve','two nostrils drilled where the tracked nostril points sit (79, 309), angled down and forward',depth_cm=0.7)
detail('lip border','Pinch','crisp vermilion edge traced along the tracked outer lip line (20 points)',radius_mm=0.6)
OUTL=[61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
BR=[[46,53,52,65,55],[276,283,282,295,285]]
def details(P,d):
    for sg,(nos,al,cr) in ((-1,(79,129,98)),(1,(309,358,327))):
        nc=(S[nos]*0.45+S[cr]*0.35+S[2]*0.2)+np.array([0,-0.12,-0.12])
        d=smaxv(d,-ellip(P,nc,np.array([0.36,0.2,0.5]),rot('x',-35)),0.1)                 # nostril
        ring=EYE_UP[0] if sg<0 else EYE_UP[1]
    if NDET>=4: d=sminv(d,chain(P,[S[i]+np.array([0,0,-0.055]) for i in OUTL],0.06),0.04)          # lip border
    return d
EYE_UP=[[33,246,161,160,159,158,157,173,133],[263,466,388,387,386,385,384,398,362]]
PARTS={'hull':('head',),'face':('head',),'neck':('head','neck','bust')}.get(STAGE,('head','neck','bust','ears'))
NDET={'d1':1}.get(STAGE,4 if STAGE=='final' else 0)
DETAILS=NDET>0 and os.environ.get('NODET') is None
NECKR=float(os.environ.get('NECKR','6.4'))
EL3=np.load('skull_ell3.npy')
EARC={-1:np.array([-EL3[3]+0.25,-1.6,EL3[2]-0.6]),1:np.array([EL3[3]-0.25,-1.6,EL3[2]-0.6])}
def sdf(P,parts=PARTS):
    rr,rv=r_lookup(P); d=(rr-rv)*0.92
    if DETAILS: d=details(P,d)
    if 'jaw' in parts: pass
    if 'neck' in parts:
        nk=capsule(P,np.array([0,-9.0,-7.6]),np.array([0,-19.0,-7.2]),NECKR,NECKR+0.5)
        d=sminv(d,nk,1.6)
    if 'bust' in parts:
        to=ellip(P,np.array([0,-26.3,-8.4]),np.array([20.0,8.0,10.5]))
        to=smaxv(to,-(P[...,1]+29.5),0.8)                         # flat base
        to=smaxv(to,np.abs(P[...,0])-17.5,1.5)                     # bust truncated at the shoulders
        for sg in (-1,1):
            to=sminv(to,capsule(P,np.array([sg*5.5,-16.0,-9.0]),np.array([sg*16.5,-21.5,-8.6]),3.4,3.0),2.0)   # trapezius to shoulder
        to=smaxv(to,np.abs(P[...,0])-17.5,1.5); to=smaxv(to,-(P[...,1]+29.5),0.8)
        d=sminv(d,to,3.0)
    if 'ears' in parts:
        for sg in (-1,1):
            Rm=rot('y',sg*-24)@rot('x',-10); c=EARC[sg]
            e=ellip(P,c+Rm@np.array([sg*0.55,0,0]),np.array([0.6,3.05,1.85]),Rm)
            rim=ellip(P,c+Rm@np.array([sg*0.8,0.15,-0.25]),np.array([0.35,2.85,1.6]),Rm)
            bowl=ellip(P,c+Rm@np.array([sg*1.15,-0.35,0.25]),np.array([0.55,1.15,0.85]),Rm)
            scapha=ellip(P,c+Rm@np.array([sg*1.1,0.7,-0.1]),np.array([0.3,1.6,1.0]),Rm)
            ear=smaxv(sminv(e,rim,0.3),-bowl,0.25); ear=smaxv(ear,-scapha,0.15)
            d=sminv(d,ear,0.35)
    return d
if __name__=='__main__':
    t0=time.time()
    xs=np.arange(-22,22+H,H); ys=np.arange(-31,14+H,H); zs=np.arange(-21,9+H,H)
    if STAGE!='final':
        pass
    vol=np.empty((len(xs),len(ys),len(zs)),np.float32)
    Y,Z=np.meshgrid(ys,zs,indexing='ij')
    for i,x in enumerate(xs):
        P=np.stack([np.full_like(Y,x),Y,Z],-1)
        vol[i]=sdf(P)
    v,f,n,_=measure.marching_cubes(vol,0,spacing=(H,H,H))
    v+=[xs[0],ys[0],zs[0]]
    pass
    print('mesh',len(v),'verts',len(f),'tris  %.0fs'%(time.time()-t0))
    np.savez_compressed('mesh3_%s.npz'%STAGE,v=v,f=f)
    if STAGE=='final': json.dump(LOG,open('log_build3.json','w'),indent=1)
    from prev import sheet
    if STAGE=='final': sheet([(v,f,(0.78,0.68,0.6))],'prev3_%s.png'%STAGE,views=((0,0),(35,-5),(90,0),(180,0)),size=420,span=36,center=(0,-7,-7))
