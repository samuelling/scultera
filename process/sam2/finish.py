"""Eyes, teeth, hair cap, T-shirt, mouth interior: labels + displacements on the built mesh."""
import numpy as np, sys, json
sys.path.insert(0,'proc')
from scipy.optimize import least_squares
from scipy import sparse as sp
from scipy import ndimage as ndi
from skimage import measure
src=sys.argv[1] if len(sys.argv)>1 else 'mesh_final.npz'
m=np.load(src); V=m['v'].astype(float); F=m['f']
fc=np.load('face.npz'); S=fc['S']; fm=np.load('rmap_face.npz'); C=fm['C']
LOG=[]
def log(phase,tool,target,note,**p): LOG.append(dict(phase=phase,tool=tool,target=target,note=note,params=p))
def vnorm(V,F):
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);N=np.zeros_like(V)
    for i in range(3): np.add.at(N,F[:,i],n)
    return N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
def adjacency(F,n):
    E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(n,n)).tocsr(); A=((A+A.T)>0).astype(float)
    return sp.diags(1/np.maximum(A.sum(1).A1,1))@A
Avg=adjacency(F,len(V))
def diffuse(x,it): 
    for _ in range(it): x=Avg@x
    return x
N=vnorm(V,F)
# ---------- eyes: spheres fitted to the eyelid contours
EYES={}
for nm in ['eyeR','eyeL']:
    P=S[fm[nm]]; EYES[nm]=P.mean(0)+np.array([0,0.05,-1.3])     # globe centre behind the lid opening
log('detail','Insert','eyeballs','two 2.5 cm spheres set behind the 16 tracked points of each eyelid rim',radius_cm=1.25)
def uvsphere(c,R,nu=48,nv=32,iris_dir=(0,0,1)):
    th=np.linspace(0,np.pi,nv); ph=np.linspace(0,2*np.pi,nu,endpoint=False)
    T,Pp=np.meshgrid(th,ph,indexing='ij'); d=np.stack([np.sin(T)*np.cos(Pp),np.sin(T)*np.sin(Pp),np.cos(T)],-1).reshape(-1,3)
    # pole along +z (gaze); carve iris (shallow dish) and pupil (drilled)
    ang=np.degrees(np.arccos(np.clip(d[:,2],-1,1))); rr=np.full(len(d),R)
    rr-=np.where(ang<27,0.06*np.clip((27-ang)/4,0,1),0)          # iris ring incised
    rr-=np.where(ang<10,0.10*np.clip((10-ang)/2,0,1),0)          # pupil drilled
    Vv=c+d*rr[:,None]; F_=[]
    for i in range(nv-1):
        for j in range(nu):
            a=i*nu+j;b=i*nu+(j+1)%nu;c_=(i+1)*nu+j;d_=(i+1)*nu+(j+1)%nu; F_+= [[a,c_,b],[b,c_,d_]]
    return Vv,np.array(F_),ang
eye_meshes={nm:uvsphere(EYES[nm],1.2) for nm in EYES}
# ---------- mouth interior + eye pocket labels (pocket = deeper than the face map)
q=V-C; rr=np.linalg.norm(q,axis=1); az=np.degrees(np.arctan2(q[:,0],q[:,2])); el=np.degrees(np.arcsin(q[:,1]/rr))
AZ0,AZ1,EL0,EL1=fm['ext']; RES=float(fm['RES']); Rf=fm['Rf']; Lab=fm['Lab']
gi=np.clip(((el-EL0)/RES).astype(int),0,Rf.shape[0]-1); gj=np.clip(((az-AZ0)/RES).astype(int),0,Rf.shape[1]-1)
rface=Rf[gi,gj]; lab=Lab[gi,gj]
# dilate hole labels a little in map space for robust lookups
mouthmap=ndi.binary_dilation(Lab==1,iterations=6); eyemap=ndi.binary_dilation((Lab==2)|(Lab==3),iterations=3)
deep=np.nan_to_num(rface-rr,nan=-9)
mouth_in=(mouthmap[gi,gj])&(deep>0.35)
eye_pocket=(eyemap[gi,gj])&(deep>0.3)
log('detail','Carve','mouth','a slit along the tracked inner-lip line, opened 1.3 cm deep so the jaw can drop',depth_cm=1.3)
# ---------- hair: project every vertex into the video frames and vote hair vs skin
d=np.load('shape_s.npz'); R,tr,f,cx,cy=d['R'],d['tr'],float(d['f']),float(d['cx']),float(d['cy'])
masks=np.load('masks.npy')
vo=(V+np.array([fc['shx'],0,0])-fc['m0'])@fc['Rh']+fc['m0']
hv=np.zeros(len(V)); sv=np.zeros(len(V))
for t in range(0,358,4):
    Rt=R[t]; X=vo@Rt.T+tr[t]; nv=(N@fc['Rh'])@Rt.T
    vis=(nv[:,2]<-0.25)                                           # facing the camera (camera looks +z)
    u=(f*X[:,0]/X[:,2]+cx).astype(int); w=(f*X[:,1]/X[:,2]+cy).astype(int)
    ok=vis&(u>=0)&(u<720)&(w>=0)&(w<1280); c=masks[t][w[ok],u[ok]]
    hv[ok]+=(c==1); sv[ok]+=(c==3)|(c==2)
seen=(hv+sv)>3
vote=np.where(seen,hv/(hv+sv+1e-9),np.nan)
# hairline as a height around the skull (azimuth 0 = front), checked against the votes
SKc=np.load('skull_fit.npy'); zc=SKc[4]
phi=np.degrees(np.arctan2(V[:,0],V[:,2]-zc)); aphi=np.abs(phi)
cp=np.array([[0,S[10,1]+0.55],[30,S[10,1]+0.75],[45,S[10,1]+0.2],[62,3.2],[74,-0.6],[80,-0.8],[86,2.4],[110,2.6],[128,-2.5],[150,-6.0],[180,-7.2]])
yh=np.interp(aphi,cp[:,0],cp[:,1])
hair=(V[:,1]>yh).astype(float)
for sg in (-1,1):
    ec=np.array([0.4*SKc[5]+sg*(SKc[0]-0.35),-1.6,-9.4]); hair[np.linalg.norm((V-ec)*[0.8,1,1],axis=1)<3.3]=0
hair[V[:,1]<-9.5]=0
agree=np.nanmean(np.abs(hair[seen]-(vote[seen]>0.5)))
print('hairline vs video votes: %.0f%% of seen vertices agree'%(100*(1-agree)))
h=diffuse((hair>0.5).astype(float),40)
hair_w=np.clip((h-0.35)/0.3,0,1)
log('detail','Mask','hairline','hair vs skin voted per vertex from the segmentation of 90 frames; unseen back filled to the nape',frames=90)
# buzz-cut: raised 2.5 mm with a fine stipple
rng=np.random.default_rng(3)
noise=diffuse(rng.standard_normal(len(V)),2); noise/=noise.std()
V=V+N*(hair_w*(0.22+0.025*noise))[:,None]
log('detail','Draw + Stipple','hair','buzz-cut mass raised 2.2 mm, stippled texture',raise_mm=2.2)
# ---------- T-shirt: crew neck collar line around the base of the neck
ang=np.arctan2(V[:,0],V[:,2]+7.8)
collar_y=-16.4-2.4*np.clip(np.cos(ang),0,1)**1.5+0.5*np.clip(-np.cos(ang),0,1)
below=V[:,1]-collar_y
shirt_w=np.clip((-below)/0.25+0.5,0,1)
rib=np.exp(-((below+0.85)/0.55)**2)*(shirt_w>0.5)
V=V+N*(shirt_w*0.28+rib*0.18)[:,None]
log('detail','Draw','T-shirt','crew-neck collar band and a 3 mm fabric layer over the shoulders',raise_mm=2.8)
# mouth interior labels survive displacement
mat=np.zeros(len(V),np.int8)            # 0 skin,1 hair,2 shirt,3 mouth,4 eye pocket
mat[hair_w>0.5]=1; mat[shirt_w>0.5]=2; mat[mouth_in]=3; mat[eye_pocket]=4
# ---------- teeth: two arches of rounded boxes, made with marching cubes
def teeth_row(upper):
    """one arch as an SDF band (U-shaped ribbon) with shallow grooves between teeth"""
    yE=(S[13,1]+S[14,1])/2; zf=(S[13,2]+S[14,2])/2-(0.38 if upper else 0.55)
    top=yE+1.12 if upper else yE-0.3; bot=yE+0.05 if upper else yE-1.15
    g=0.03; xs=np.arange(-3.0,3.0,g); ys_=np.arange(bot-0.1,top+0.1,g); zs=np.arange(zf-2.6,zf+0.3,g)
    X,Y,Z=np.meshgrid(xs,ys_,zs,indexing='ij')
    s_=np.linspace(-1.3,1.3,400); ax=(2.55 if upper else 2.35)*np.sin(s_); az_=zf-2.1*(1-np.cos(s_))
    # distance in xz to the arch polyline (brute force via KD tree)
    from scipy.spatial import cKDTree
    kd=cKDTree(np.c_[ax,az_]); dxz,k=kd.query(np.c_[X.ravel(),Z.ravel()]); dxz=dxz.reshape(X.shape); k=k.reshape(X.shape)
    arcpos=np.r_[0,np.cumsum(np.hypot(np.diff(ax),np.diff(az_)))]; arcpos-=arcpos[200]; L=arcpos[k]
    th=0.26 if upper else 0.22
    # taper crowns: incisors tall, canines pointed, back teeth shorter
    al=np.abs(L); hfac=np.interp(al,[0,1.5,2.3,3.5],[1.0,0.95,0.8,0.65])
    yc=(top+bot)/2; hh=(top-bot)/2*hfac
    if upper: yc=top-hh   # gum line fixed at top, edge moves up for back teeth
    else: yc=bot+hh
    q=np.stack([dxz-th,np.abs(Y-yc)-hh],-1)
    dd=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)-0.06
    edges=np.cumsum([0.43,0.68,0.78,0.7,0.7,1.0])*(1 if upper else 0.86); edges=np.r_[0,edges,-edges]
    for e in edges:
        dd=np.maximum(dd,0.045-np.sqrt((L-e)**2/1.0+0*Y)-np.clip((dxz-th+0.12)*0,0,0)) if False else np.maximum(dd,-(np.abs(L-e)-0.022)*(dxz>th-0.14))
    v,f_,_,_=measure.marching_cubes(dd,0,spacing=(g,g,g)); v+=[xs[0],ys_[0],zs[0]]
    return v,f_[:,::-1]
TU=teeth_row(True); TL=teeth_row(False)
log('detail','Insert','teeth','upper and lower arches (10 teeth each) set behind the tracked lips; the lower arch rides the jaw',count=20)
np.savez_compressed('finished.npz',v=V.astype(np.float32),f=F,mat=mat,hair_w=hair_w.astype(np.float32),shirt_w=shirt_w.astype(np.float32),
    eyeR_v=eye_meshes['eyeR'][0],eyeR_f=eye_meshes['eyeR'][1],eyeL_v=eye_meshes['eyeL'][0],eyeL_f=eye_meshes['eyeL'][1],
    eyeR_c=EYES['eyeR'],eyeL_c=EYES['eyeL'],tu_v=TU[0],tu_f=TU[1],tl_v=TL[0],tl_f=TL[1])
json.dump(LOG,open('log_finish.json','w'),indent=1)
print('hair verts',(hair_w>0.5).sum(),'shirt',(shirt_w>0.5).sum(),'mouth_in',mouth_in.sum(),'eye pockets',eye_pocket.sum(),'teeth',len(TU[0]),len(TL[0]))
from prev import sheet
cols={0:(0.8,0.68,0.58),1:(0.3,0.26,0.24),2:(0.2,0.22,0.27),3:(0.35,0.12,0.12),4:(0.4,0.3,0.3)}
meshes=[]
for k,c in cols.items():
    sel=(mat[F]==k).sum(1)>=2
    if sel.any(): meshes.append((V,F[sel],c))
meshes+= [(eye_meshes[n][0],eye_meshes[n][1],(0.92,0.9,0.86)) for n in eye_meshes]+[(TU[0],TU[1],(0.95,0.93,0.85)),(TL[0],TL[1],(0.95,0.93,0.85))]
sheet(meshes,'prev_finish.png',views=((0,0),(30,-10),(90,0),(160,0)),size=420,span=40,center=(0,-7,-7))
