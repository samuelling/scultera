"""Angular (radial) maps around a head centre: face surface r_face, hull r_hull."""
import numpy as np, sys
sys.path.insert(0,'proc')
from meshutil import boundary_loops, loop_subdivide
from scipy import ndimage as ndi
fc=np.load('face.npz'); S=fc['S']; F=fc['F']
MOUTH=[78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
EYER=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
EYEL=[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
cut=np.zeros(len(F),bool)
for ring in (MOUTH,EYER,EYEL): cut|=np.isin(F,ring).all(1)
F=F[~cut]; print('opened',cut.sum(),'faces across eyes and mouth')
C=np.array([0.0,0.0,-8.0]); RES=0.2   # degrees
AZ0,AZ1,EL0,EL1=-180,180,-89,89
naz=int((AZ1-AZ0)/RES); nel=int((EL1-EL0)/RES)
def sph(P):
    q=P-C; r=np.linalg.norm(q,axis=-1); az=np.degrees(np.arctan2(q[...,0],q[...,2])); el=np.degrees(np.arcsin(q[...,1]/r)); return az,el,r
def dirs():
    az=np.radians(AZ0+RES*(np.arange(naz)+0.5)); el=np.radians(EL0+RES*(np.arange(nel)+0.5))
    A,E=np.meshgrid(az,el,indexing='xy')     # (nel,naz)
    return np.stack([np.cos(E)*np.sin(A),np.sin(E),np.cos(E)*np.cos(A)],-1)
# --- hole fills (eyes, mouth) as labelled fans
V=S[:468].copy(); loops=sorted(boundary_loops(F),key=len)
lab=np.zeros(len(F),int); Fl=[F]; Ll=[lab]; names={}
outer=max(boundary_loops(F),key=lambda L: len(L))
holes=[L for L in boundary_loops(F) if L is not outer and len(L)!=len(outer)]
for L in holes:
    c=V[L].mean(0); k=len(V); V=np.r_[V,[c]]
    nm='mouth' if c[1]<-5 else ('eyeL' if c[0]>0 else 'eyeR'); code={'mouth':1,'eyeR':2,'eyeL':3}[nm]; names[nm]=L
    fan=np.array([[L[(i+1)%len(L)],L[i],k] for i in range(len(L))])
    # orientation: match neighbours (normal toward +z roughly)
    n=np.cross(V[fan[0,1]]-V[fan[0,0]],V[fan[0,2]]-V[fan[0,0]])
    if n[2]<0: fan=fan[:,[1,0,2]]
    Fl.append(fan); Ll.append(np.full(len(fan),code))
F2=np.concatenate(Fl); L2=np.concatenate(Ll)
# carry landmark weights through subdivision (for expression transfer): one-hot over 468 landmarks + fill vertices
nW=len(V); Wt=np.eye(nW)
Vs,Fs=V,F2; Ls=L2; Ws=Wt
for lvl in range(3):
    Vs,Fs_new,(Ws,)=loop_subdivide(Vs,Fs,(Ws,)); Ls=np.tile(Ls,4); Fs=Fs_new
print('face mesh', len(Vs),'verts', len(Fs),'tris; holes',{k:len(v) for k,v in names.items()},'outer',len(outer))
# fill vertices' weights -> spread to their hole loop (so deltas interpolate from the rim)
for nm,L in names.items():
    k=[i for i in range(468,nW) if np.allclose(V[i],S[L].mean(0))][0]
    Ws[:,L]+=Ws[:,[k]]/len(L); Ws[:,k]=0
Ws=Ws[:,:468]
# --- rasterize r_face in angular space (z-buffer: keep largest r = outermost)
az,el,r=sph(Vs); gx=(az-AZ0)/RES-0.5; gy=(el-EL0)/RES-0.5
Rf=np.full((nel,naz),np.nan); Tid=np.full((nel,naz),-1); B=np.zeros((nel,naz,3)); Lab=np.zeros((nel,naz),int)
for t,(a,b,c) in enumerate(Fs):
    xs=gx[[a,b,c]];ys=gy[[a,b,c]]
    x0,x1=int(np.floor(xs.min())),int(np.ceil(xs.max()));y0,y1=int(np.floor(ys.min())),int(np.ceil(ys.max()))
    X,Y=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1))
    den=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
    if abs(den)<1e-12: continue
    l0=((ys[1]-ys[2])*(X-xs[2])+(xs[2]-xs[1])*(Y-ys[2]))/den; l1=((ys[2]-ys[0])*(X-xs[2])+(xs[0]-xs[2])*(Y-ys[2]))/den; l2=1-l0-l1
    m=(l0>=-1e-6)&(l1>=-1e-6)&(l2>=-1e-6)
    if not m.any(): continue
    X,Y,l0,l1,l2=X[m],Y[m],l0[m],l1[m],l2[m]; rr=l0*r[a]+l1*r[b]+l2*r[c]
    cur=Rf[Y,X]; up=np.isnan(cur)|(rr>cur)
    Rf[Y[up],X[up]]=rr[up]; Tid[Y[up],X[up]]=t; B[Y[up],X[up]]=np.c_[l0,l1,l2][up]; Lab[Y[up],X[up]]=Ls[t]
print('face covers %.1f%% of the map'%(100*np.isfinite(Rf).mean()))
np.savez_compressed('rmap_face.npz',Rf=Rf,Tid=Tid,B=B,Lab=Lab,Vs=Vs,Fs=Fs,Ws=Ws.astype(np.float32),C=C,RES=RES,ext=[AZ0,AZ1,EL0,EL1],
    eyeR=names['eyeR'],eyeL=names['eyeL'],mouth=names['mouth'],outer=outer)
