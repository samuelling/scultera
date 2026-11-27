"""Take 4, step 6: hair (raised where the frames say hair), T-shirt, mouth slit, teeth, and eyeballs painted from the frontal frame."""
import numpy as np, cv2, sys, json, re
sys.path.insert(0,'proc'); from common import *
import sdf4
from scipy import sparse as sp, ndimage as ndi
from skimage import measure
m=np.load('mesh4_final.npz'); V=m['v'].astype(float); F=m['f']; S=fc['S']; fm=sdf4.fm; C=sdf4.C
def vnorm(V,F):
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);N=np.zeros_like(V)
    for i in range(3): np.add.at(N,F[:,i],n)
    return N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
N=vnorm(V,F)
if (N[np.argmax(V[:,1])][1])<0: F=F[:,::-1].copy(); N=-N
E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); A=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(len(V),)*2).tocsr(); A=((A+A.T)>0).astype(float); Avg=sp.diags(1/np.maximum(A.sum(1).A1,1))@A
rr,az,el=sdf4.angles(V)
hm=np.load('work/hairmap.npy'); TH,TW=hm.shape
tex=lambda M,order=1: ndi.map_coordinates(M,[(90-el)/180*TH-0.5,(az+180)/360*TW-0.5],order=order,mode='nearest')
hair_w=np.clip((tex(hm)-0.4)/0.25,0,1); top=np.clip((el-np.where(np.abs(az)<55,49,40))/6,0,1); hair_w=np.maximum(hair_w,top*top*(3-2*top))   # seen from above, the buzz cut reads as skin to the segmenter
_g=cv2.GaussianBlur(cv2.imread('texture.jpg',0).astype(np.float32),(0,0),5); lum=tex(_g); med=np.median(lum[(np.abs(az)<40)&(el>-30)&(el<20)])
dk=np.clip((0.72-lum/med)/0.2,0,1)*np.clip((el-30)/8,0,1); hair_w=np.maximum(hair_w,dk)        # and where the paint itself is hair-dark above the brow
hair_w[V[:,1]<-9.0]=0; hair_w[(np.abs(az)<60)&(el<12)]=0
rng=np.random.default_rng(3); noise=Avg@(Avg@rng.standard_normal(len(V))); noise/=noise.std()
V=V+N*(hair_w*(0.16+0.03*noise))[:,None]
ang=np.arctan2(V[:,0],V[:,2]+8.6); collar=-15.6-2.6*np.clip(np.cos(ang),0,1)**1.5+0.6*np.clip(-np.cos(ang),0,1); below=V[:,1]-collar
shirt_w=np.clip((-below)/0.25+0.5,0,1); rib=np.exp(-((below+0.85)/0.55)**2)*(shirt_w>0.5); V=V+N*(shirt_w*0.25+rib*0.18)[:,None]
Rf=fm['Rf']; Lab=fm['Lab']; deep=np.nan_to_num(sdf4.fmap(np.nan_to_num(Rf,nan=0).astype(np.float32),az,el,0)-rr,nan=-9)
labv=sdf4.fmap(ndi.grey_dilation(Lab,size=9).astype(np.float32),az,el,0)
mat=np.zeros(len(V),np.int8); mat[hair_w>0.5]=1; mat[shirt_w>0.5]=2; mat[(labv==1)&(deep>0.4)&(el<-10)&(np.abs(az)<40)]=3; mat[(labv>=2)&(deep>0.35)&(el>-10)&(np.abs(az)<50)]=4
# eyeballs, finer than before, painted from the frontal key frame through that frame's own eyelid outline
tj=json.load(open('work/texture.json')); tf=tj['front']; wb=np.array(tj['wb']); p=np.load('poses4.npz'); lm=np.load(B3+'track.npz')['lm'][tf]*1.5
img=cv2.imread('full/f%04d.jpg'%(tf+1)).astype(np.float32)
RINGS={'eyeR':[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],'eyeL':[263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]}
def uvsphere(c,R,nu=72,nv=56):
    th=np.linspace(0,np.pi,nv)**1.25/np.pi**0.25; ph=np.linspace(0,2*np.pi,nu,endpoint=False); T,Pp=np.meshgrid(th,ph,indexing='ij')
    d=np.stack([np.sin(T)*np.cos(Pp),np.sin(T)*np.sin(Pp),np.cos(T)],-1).reshape(-1,3); Fq=[]
    for i in range(nv-1):
        for j in range(nu): a=i*nu+j;b=i*nu+(j+1)%nu;c_=(i+1)*nu+j;d_=(i+1)*nu+(j+1)%nu; Fq+=[[a,c_,b],[b,c_,d_]]
    return c+d*R,np.array(Fq),d
out={}
for nm,ring in RINGS.items():
    c=S[fm[nm]].mean(0)+np.array([0,0.05,-1.3]); Ve,Fe,d=uvsphere(c,1.2); uv,_=proj(toold(Ve),p['R'][tf],p['tr'][tf],1.5)
    msk=np.zeros(img.shape[:2],np.uint8); cv2.fillPoly(msk,[np.round(lm[ring,:2]).astype(np.int32)],1); msk=cv2.erode(msk,np.ones((3,3),np.uint8))
    ui=np.clip(np.round(uv[:,0]).astype(int),0,img.shape[1]-1); vi=np.clip(np.round(uv[:,1]).astype(int),0,img.shape[0]-1)
    lin=np.clip((cv2.GaussianBlur(img,(0,0),0.8)[vi,ui]/255)**2.2*wb,0,1)[:,::-1]; inm=(msk[vi,ui]>0)&(d[:,2]>0.25)
    px=np.clip((img[msk>0]/255)**2.2*wb,0,1)[:,::-1]; scl=np.percentile(px,80,axis=0)*np.array([1.0,0.98,0.95])
    col=np.where(inm[:,None],lin,scl); col=np.where((d[:,2]<0.6)[:,None]&~inm[:,None],scl*0.9,col)
    out[nm+'_v']=Ve.astype(np.float32); out[nm+'_f']=Fe; out[nm+'_c']=c; out[nm+'_col']=col.astype(np.float32)
    print(nm,'painted verts',int(inm.sum()),'of',len(Ve),' sclera',scl.round(2))
src=open(B3+'proc/finish.py').read(); exec(src[src.index('def teeth_row(upper):'):src.index('TU=teeth_row(True)')])
TU=teeth_row(True); TL=teeth_row(False)
np.savez_compressed('finished4.npz',v=V.astype(np.float32),f=F,mat=mat,hair_w=hair_w.astype(np.float32),shirt_w=shirt_w.astype(np.float32),tu_v=TU[0],tu_f=TU[1],tl_v=TL[0],tl_f=TL[1],**out)
print('verts',len(V),'hair',int((mat==1).sum()),'shirt',int((mat==2).sum()),'mouth',int((mat==3).sum()),'eye pockets',int((mat==4).sum()))
