"""Paint layer: project the video onto the posed + expression-matched sculpt, frame by frame, visibility-tested."""
import numpy as np, json
from PIL import Image
from scipy import sparse as sp
from scipy import ndimage as ndi
fin=np.load('finished.npz'); V0=fin['v'].astype(float); F=fin['f']; mat=fin['mat']
fc=np.load('face.npz'); D=fc['D']; Wv=sp.load_npz('Wv.npz'); d=np.load('shape_s.npz')
R,tr,f,cx,cy=d['R'],d['tr'],float(d['f']),float(d['cx']),float(d['cy']); masks=np.load('masks.npy')
def vnorm(V):
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);N=np.zeros_like(V)
    for i in range(3): np.add.at(N,F[:,i],n)
    return N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
def toold(P): return (P+np.array([fc['shx'],0,0])-fc['m0'])@fc['Rh']+fc['m0']
acc=np.zeros((len(V0),3)); wsum=np.zeros(len(V0))
mean_prev=None
for PASS in (0,1):
  acc[:]=0; wsum[:]=0
  for t in range(0,358,2):
      V=V0+Wv@D[t,:468]; X=toold(V)@R[t].T+tr[t]; N=vnorm(V)@fc['Rh']@R[t].T
      u=f*X[:,0]/X[:,2]+cx; w=f*X[:,1]/X[:,2]+cy
      ui=np.round(u).astype(int); wi=np.round(w).astype(int); ok=(ui>=1)&(ui<719)&(wi>=1)&(wi<1279)
      # depth buffer from all mesh vertices (splat 3x3)
      zb=np.full((1280,720),1e9)
      for du in (-1,0,1):
          for dw in (-1,0,1):
              np.minimum.at(zb,(np.clip(wi[ok]+dw,0,1279),np.clip(ui[ok]+du,0,719)),X[ok,2])
      vis=ok.copy(); vis[ok]=X[ok,2]<=zb[wi[ok],ui[ok]]+0.25
      view=-(X/np.linalg.norm(X,axis=1,keepdims=True)); facing=(N*view).sum(1)
      vis&=facing>0.3
      img=np.asarray(Image.open('frames/f%04d.jpg'%(t+1)),float)/255
      cls=masks[t].copy(); person=ndi.binary_erosion(np.isin(cls,[1,2,3]),iterations=5); shirt=ndi.binary_erosion(cls==4,iterations=4); cls=np.where(person,cls,np.where(shirt,4,0))
      c=cls[wi[vis],ui[vis]]; want=np.where(mat[vis]==2,c==4,np.isin(c,[1,2,3]))
      idx=np.where(vis)[0][want]
      col=img[wi[idx],ui[idx]]**2.2
      wt=facing[idx]**3*(np.maximum(np.exp(-((col-mean_prev[idx])**2).sum(1)/0.004),0.05) if mean_prev is not None else 1)
      acc[idx]+=col*wt[:,None]; wsum[idx]+=wt
  mean_prev=np.where(wsum[:,None]>0.05,acc/np.maximum(wsum[:,None],1e-9),0)
seen=wsum>0.05
C=np.zeros((len(V0),3)); C[seen]=acc[seen]/wsum[seen][:,None]
print('painted %.0f%% of vertices from the video'%(100*seen.mean()))
E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(len(V0),)*2).tocsr(); A=((A+A.T)>0).astype(float)
Avg=sp.diags(1/np.maximum(A.sum(1).A1,1))@A
# de-lighting on the seen skin: the window lit his right cheek, the lamp his left; flatten the broad light, keep detail
skin=(mat==0)&seen; L=C.mean(1)+1e-4; num=np.where(skin,L,0.0); den=skin.astype(float)
for it in range(300): num=Avg@num; den=Avg@den
Ls=num/np.maximum(den,1e-6); Lt=np.median(L[skin])
chroma=C/L[:,None]; cmed=np.median(chroma[skin],0)
over=np.clip((Ls/Lt-1.05)/0.6,0,1)[:,None]
chroma=chroma*(1-0.75*over)+cmed*0.75*over
C=np.where(skin[:,None],np.clip(chroma*(Lt*L/Ls)[:,None],0,1),C)
# fill unseen: diffuse from seen within each material, else material median
E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(len(V0),)*2).tocsr(); A=((A+A.T)>0).astype(float)
Avg=sp.diags(1/np.maximum(A.sum(1).A1,1))@A
med={k:np.median(C[seen&(mat==k)&(np.isin(mat,[0]) | True)],0) if (seen&(mat==k)).sum()>50 else None for k in range(5)}
med[3]=np.array([0.06,0.015,0.012]); med[4]=np.array([0.05,0.03,0.025])
base=np.array([med[k] if med[k] is not None else med[0] for k in mat])
known=seen.astype(float); Cf=np.where(seen[:,None],C,base); kw=known.copy()
for it in range(400):
    Cn=Avg@(Cf*kw[:,None]); kn=Avg@kw
    upd=~seen; Cf[upd]=np.where(kn[upd,None]>1e-4,Cn[upd]/np.maximum(kn[upd,None],1e-6),Cf[upd]); kw[upd]=np.maximum(kw[upd],kn[upd]*0.995)
# far from any seen vertex -> material base
bl=np.clip(kw/0.05,0,1)[:,None]; Cf=Cf*bl+base*(1-bl)
Cf[mat==3]=med[3]; Cf[mat==4]=med[4]
# hair: dark buzz cut over a little scalp; neck below the jaw evened out
hairc=np.array([0.022,0.018,0.016]); hw=fin['hair_w'][:,None]
rng=np.random.default_rng(5); hn=Avg@(Avg@rng.standard_normal(len(V0))); hn=(hn/hn.std())[:,None]
Cf=Cf*(1-0.92*hw)+hairc*(1+0.25*hn)*0.92*hw
neck=(V0[:,1]<-10.5)&(mat==0)&(V0[:,2]<-0.5)
Cs=Cf.copy()
for it in range(40): Cs=Avg@Cs
jawd=np.clip((-10.5-V0[:,1])/3.0,0,1)
nb=(jawd*neck)[:,None]
for it in range(80): Cs=Avg@Cs
Cf=Cf*(1-nb)+(0.4*Cs+0.6*med[0]*0.9)*nb
# ears: the lamp behind him shines through them; tone the glow down
for sgn in (-1,1):
    ec=np.array([0.4*np.load('skull_fit.npy')[5]+sgn*(np.load('skull_fit.npy')[0]-0.35),-1.6,-9.4])
    ew=np.clip(1-(np.linalg.norm(V0-ec,axis=1)-2.6)/1.2,0,1)[:,None]
    g=Cf.mean(1,keepdims=True); Cf=Cf*(1-0.55*ew)+(g*np.array([1.15,0.95,0.85])*0.85)*0.55*ew
np.save('paint.npy',Cf.astype(np.float32))
json.dump(dict(painted=float(seen.mean()),frames=179),open('paint_stats.json','w'))
