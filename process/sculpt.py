import numpy as np, time
from scipy.spatial import cKDTree
import scipy.sparse as sps
t0=time.time()
D=np.load('base.npz');V=D['v'].astype(np.float64);Fc=D['f'];E=D['eyes']
rng=np.random.default_rng(4)
def normals(V):
    fn=np.cross(V[Fc[:,1]]-V[Fc[:,0]],V[Fc[:,2]]-V[Fc[:,0]])
    vn=np.zeros_like(V)
    for i in range(3): np.add.at(vn,Fc[:,i],fn)
    return vn/(np.linalg.norm(vn,axis=1,keepdims=True)+1e-12)
e=np.vstack([Fc[:,[0,1]],Fc[:,[1,2]],Fc[:,[2,0]]])
A=sps.coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(V),len(V))).tocsr();A=((A+A.T)>0).astype(float)
A=sps.diags(1/np.asarray(A.sum(1)).ravel())@A
def fall(t): t=np.clip(t,0,1);return 1-(3*t*t-2*t**3)
class Brush:
    def __init__(s): s.N=normals(V);s.T=cKDTree(V)
    def refresh(s): s.N=normals(V);s.T=cKDTree(V)
    def dab_draw(s,c,R,st):
        i=np.array(s.T.query_ball_point(c,R),int)
        if len(i)==0: return
        w=fall(np.linalg.norm(V[i]-c,axis=1)/R);V[i]+=s.N[i]*(st*w)[:,None]
    def dab_crease(s,c,R,st,pinch=0.4):
        i=np.array(s.T.query_ball_point(c,R),int)
        if len(i)==0: return
        d=V[i]-c;w=fall(np.linalg.norm(d,axis=1)/R);n=s.N[i]
        tang=d-(d*n).sum(1)[:,None]*n
        V[i]+=-n*(st*w)[:,None]-tang*(pinch*w)[:,None]*0.5
    def grab(s,c,R,delta):
        w=fall(np.linalg.norm(V-c,axis=1)/R);V[:]+=w[:,None]*np.array(delta)
    def smooth(s,c,R,lam,it=1,mask=None):
        if mask is None:
            i=np.array(s.T.query_ball_point(c,R),int);w=np.zeros(len(V));w[i]=fall(np.linalg.norm(V[i]-c,axis=1)/R)
        else: w=mask
        for _ in range(it):
            L=A@V-V;V[:]+=L*(lam*w)[:,None]
    def stroke(s,kind,pts,R,st,spacing=0.3,**kw):
        pts=np.array(pts,float);P=[pts[0]]
        for a,b in zip(pts[:-1],pts[1:]):
            n=max(1,int(np.linalg.norm(b-a)/(R*spacing)))
            P+= [a+(b-a)*t for t in np.linspace(0,1,n+1)[1:]]
        for p in P:
            _,j=s.T.query(p);c=V[j]
            getattr(s,'dab_'+kind)(c,R,st,**kw)
        s.refresh()
B=Brush()
# 0. global smooth to relax voxel stepping
B.smooth(None,None,0.5,it=3,mask=np.ones(len(V)))
B.refresh()
# 1. Grab: droop the head down and turn it slightly toward camera
B.grab(np.array([-4.6,0,0.45]),2.0,(0.0,-0.14,-0.10))
B.grab(np.array([-4.9,0,0.35]),0.9,(0.0,-0.03,-0.06))
# 2. Flatten belly / chin onto the floor
z=V[:,2];V[:,2]=np.where(z<0.09,0.035+0.055*(1/(1+np.exp(-(z-0.035)/0.02))),z)
low=np.clip((0.12-V[:,2])/0.1,0,1);B.smooth(None,None,0.5,it=4,mask=low);B.refresh()
np.savez('stage_grab.npz',v=V.astype(np.float32),f=Fc)
# 3. Mouth crease, slightly downturned at the corner (both sides)
for sg in (-1,1):
    xs=np.linspace(-4.98,-2.9,30);zm=np.interp(xs,[-4.98,-4.3,-3.5,-3.15,-2.9],[0.22,0.30,0.36,0.33,0.25])
    pts=[];
    for xv,zv in zip(xs,zm):
        cand=np.where((np.abs(V[:,0]-xv)<0.03)&(np.abs(V[:,2]-zv)<0.03)&(np.sign(V[:,1])==sg))[0]
        if len(cand): j=cand[np.argmax(np.abs(V[cand,1]))];pts.append(V[j])
    B.stroke('crease',pts,0.07,0.028,spacing=0.25,pinch=0.5)
    B.stroke('draw',[p+np.array([0,0,0.06]) for p in pts[2:-4]],0.06,0.008,spacing=0.3)
# 4. Nostrils, ear openings, brow ridges, neck folds
for sg in (-1,1):
    B.stroke('crease',[np.array([-4.93,sg*0.16,0.40])],0.05,0.035,pinch=0.2)
    B.stroke('crease',[np.array([-2.78,sg*0.86,0.78]),np.array([-2.76,sg*0.86,0.70])],0.075,0.05,pinch=0.25)
    c,o,up,bk=E[0 if sg<0 else 1].reshape(4,3)
    arc=[c+o*0.2+up*0.30+bk*t for t in np.linspace(-0.3,0.25,8)]
    B.stroke('draw',arc,0.12,0.012,spacing=0.3)
    for xf in (-2.62,-2.40,-2.18):
        B.stroke('crease',[np.array([xf,sg*0.95,0.95]),np.array([xf+0.05,sg*1.0,0.45])],0.06,0.014,pinch=0.3)
    # armpit/hip folds
    B.stroke('crease',[np.array([-1.45,sg*1.15,0.55]),np.array([-1.55,sg*1.05,0.2])],0.07,0.015,pinch=0.3)
    B.stroke('crease',[np.array([0.95,sg*1.1,0.55]),np.array([1.05,sg*1.0,0.2])],0.07,0.015,pinch=0.3)
# 5. tail rings
for xr in np.arange(2.3,5.1,0.3):
    yc=np.interp(xr,[2,2.8,3.6,4.4,5,5.35],[0,-0.08,-0.3,-0.62,-0.92,-1.12])
    ring=np.where(np.abs(V[:,0]-xr)<0.02)[0];ring=ring[V[ring,2]>0.15]
    ring=ring[np.argsort(np.arctan2(V[ring,2]-0.3,V[ring,1]-yc))][::6]
    for j in ring: B.dab_crease(V[j].copy(),0.07,0.008,pinch=0.25)
B.refresh()
np.savez('stage_crease.npz',v=V.astype(np.float32),f=Fc)
# 6. dorsal mask
N=B.N
eyed=np.min([np.linalg.norm(V-E[k][:3],axis=1) for k in range(2)],axis=0)
dors=np.clip((N[:,2]+0.15)/0.5,0,1)*np.clip((V[:,2]-0.12)/0.2,0,1)
# 7. tubercles layer (sculpted bumps)
cand=np.where((dors>0.3)&(eyed>0.42))[0]
nb=9000;ci=rng.choice(cand,nb,replace=False);C=V[ci].copy()
head=C[:,0]<-2.55;limb=(np.abs(C[:,1])>1.35)&(C[:,0]<2.2)
rr=rng.uniform(0.045,0.085,nb);rr[head]*=0.6;rr[limb]*=0.6
hh=rr*rng.uniform(0.32,0.5,nb)*dors[ci]
T=cKDTree(C);dist,idx=T.query(V,k=6)
disp=np.zeros(len(V))
for kk in range(6):
    r_=rr[idx[:,kk]];disp=np.maximum(disp,hh[idx[:,kk]]*fall(dist[:,kk]/r_))
# fine scale layer everywhere except eyes
nf=60000;cf=V[rng.choice(len(V),nf,replace=False)];T2=cKDTree(cf);d2,_=T2.query(V)
disp+=0.006*fall(d2/0.03)*(eyed>0.36)
V+=N*disp[:,None]
B.smooth(None,None,0.3,it=1,mask=np.ones(len(V))*0.5)
N=normals(V)
# 8. colour: leopard gecko pattern (vertex colours)
dors=np.clip((N[:,2]+0.25)/0.55,0,1)*np.clip((V[:,2]-0.08)/0.25,0,1)
belly=np.array([0.94,0.89,0.78]);yel=np.array([0.93,0.70,0.20]);top=np.array([0.86,0.56,0.12])
hi=np.clip((V[:,2]-0.6)/0.6,0,1)[:,None]
col=belly+(yel-belly)*dors[:,None];col=col+(top-col)*hi*dors[:,None]
# tail: more saturated, slight lavender-gray underside band (common in leopard geckos)
tl=np.clip((V[:,0]-1.8)/0.4,0,1)[:,None]
col=col*(1-0.35*tl*(1-dors[:,None]))+np.array([0.62,0.58,0.68])*0.35*tl*(1-dors[:,None])
# tubercle tips slightly paler
col+= (np.clip(disp/0.03,0,1)*0.12)[:,None]*np.array([1,0.95,0.8])
# spots
sc=np.where(dors>0.45)[0];ns=1300;si=rng.choice(sc,ns,replace=False);S=V[si]
srad=rng.uniform(0.05,0.13,ns);srad[S[:,0]<-2.55]*=0.6;srad[np.abs(S[:,1])>1.3]*=0.55
T3=cKDTree(S);d3,i3=T3.query(V,k=3)
spot=np.zeros(len(V))
noise=np.sin(V[:,0]*37+V[:,1]*21)*np.sin(V[:,1]*29-V[:,2]*41)*0.25+1
for kk in range(3):
    t=d3[:,kk]/(srad[i3[:,kk]]*noise)
    spot=np.maximum(spot,np.clip((1-t)/0.25,0,1))
spot*=np.clip(dors*1.6-0.3,0,1)
dark=np.array([0.10,0.07,0.05])
col=col*(1-spot[:,None]*0.92)+dark*spot[:,None]*0.92
col=np.clip(col,0,1)
col=np.where(col<=0.04045,col/12.92,((col+0.055)/1.055)**2.4)
np.savez('sculpt.npz',v=V.astype(np.float32),f=Fc,col=col.astype(np.float32),eyes=E)
print('done',time.time()-t0)
