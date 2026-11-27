"""Take 4, step 5: one sharp texture instead of an average of 249 blurry projections.
A handful of slow, calm key frames are projected onto the surface, each given that frame's expression first;
later views are colour-matched and flow-registered to what is already painted, then blended, best view winning."""
import numpy as np, cv2, sys, json
sys.path.insert(0,'proc'); from common import *
import sdf4
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as Rot
PAD=96; TW0,TH=2048,1024; TW=TW0+2*PAD        # padded past +-180 deg so every filter wraps cleanly round the back
az=-180+(np.arange(TW)-PAD+0.5)*360/TW0; el=90-(np.arange(TH)+0.5)*180/TH; D=sdf4.dirs(az,el); C=sdf4.C
try: r0=np.load('work/posmap_r.npy')
except FileNotFoundError: r0=sdf4.trace(D,parts=('head','face','neck','details')); np.save('work/posmap_r.npy',r0.astype(np.float32))
P0=C+r0[...,None]*D
nrm=np.cross(np.gradient(P0,axis=1),np.gradient(P0,axis=0)); nrm/=np.maximum(np.linalg.norm(nrm,axis=-1,keepdims=True),1e-9); nrm*=np.sign((nrm*D).sum(-1,keepdims=True)+1e-9)
p=np.load('poses4.npz'); R,tr,src=p['R'],p['tr'],p['src']; masks=np.load(B3+'masks.npy',mmap_mode='r')
good=sh['good']; idx=sh['idx']; Dx=fc['D']; kof={int(t):k for k,t in enumerate(idx) if good[k]}
# expression of a frame as a displacement of every texel (the same transfer the animation uses)
fm=sdf4.fm; Tid=fm['Tid']; Bc=fm['B']; Fs=fm['Fs']; Ws=fm['Ws']; ins=Tid>=0
dout,(ii,jj)=ndi.distance_transform_edt(~ins,return_indices=True); dec=np.exp(-((dout*sdf4.RES)/7.0)**2).astype(np.float32)
A2,E2=np.meshgrid(az,el); A2w=((A2+180)%360)-180
def expr(t):
    if t not in kof: return 0
    dV=Ws@Dx[kof[t],:468]; cell=(Bc[...,None]*dV[Fs[np.maximum(Tid,0)]]).sum(2); cell=cell[ii,jj]*dec[...,None]
    return np.stack([sdf4.fmap(cell[...,c].astype(np.float32),A2w,E2) for c in range(3)],-1)
cam=lambda t: tonew((-R[t].T@tr[t])[None])[0]
def vdir(t): v=cam(t)-C; return v/np.linalg.norm(v)
MOV=[13,14,17,0,61,291,152,159,145,386,374,105,334]
calm=lambda t: float(np.abs(Dx[kof[t],MOV]).mean()) if t in kof else 0.25
def speed(t): a,b=max(t-1,0),min(t+1,len(R)-1); return np.degrees(np.linalg.norm(Rot.from_matrix(R[b]@R[a].T).as_rotvec()))/max(b-a,1)
def pick(taz,tel,cands,extra=lambda t:0):
    tv=sdf4.dirs(np.array([taz]),np.array([tel]))[0,0]
    return min(cands,key=lambda t: np.degrees(np.arccos(np.clip(vdir(t)@tv,-1,1)))+40*calm(t)+3*speed(t)+extra(t))
tracked=[t for t in range(len(R)) if t in kof]; back=[t for t in list(range(105,298))+list(range(409,550))]
S=fc['S']; ap=lambda t: -8*float((S[159,1]+Dx[kof[t],159,1]-S[145,1]-Dx[kof[t],145,1])+(S[386,1]+Dx[kof[t],386,1]-S[374,1]-Dx[kof[t],374,1]))   # open eyes preferred
views=[('front',pick(0,8,tracked,lambda t: 0.35*ap(t)+8*speed(t)))]
for a_ in (38,-38,72,-72): views.append(('side %+d'%a_,pick(a_,6,tracked)))
views.append(('from below',pick(0,-30,tracked))); views.append(('from above',pick(0,50,tracked)))
for a_ in (105,-105,130,-130,155,180): views.append(('back %+d'%a_,pick(a_,5,back)))
print('key frames:',[(n,t,'%.1fs'%(t/30)) for n,t in views])
def project(t):
    Pt=P0+expr(t); uv,z=proj(toold(Pt.reshape(-1,3)),R[t],tr[t],1.5); uv=uv.reshape(TH,TW,2).astype(np.float32); z=z.reshape(TH,TW)
    img=cv2.imread('full/f%04d.jpg'%(t+1)); col=cv2.remap(img,uv[...,0],uv[...,1],cv2.INTER_LINEAR).astype(np.float32)
    # visibility: depth buffer of the whole surface in this view
    ui=np.round(uv[...,0]/2).astype(int); vi=np.round(uv[...,1]/2).astype(int); ok=(ui>=1)&(ui<539)&(vi>=1)&(vi<959)&(r0>1.2)
    zb=np.full((960,540),np.inf)
    for du in (-1,0,1):
        for dv in (-1,0,1): np.minimum.at(zb,(vi[ok]+dv,ui[ok]+du),z[ok])
    vis=ok.copy(); vis[ok]=z[ok]<=zb[vi[ok],ui[ok]]+0.5
    v=cam(t)-Pt; v/=np.linalg.norm(v,axis=-1,keepdims=True); fa=(nrm*v).sum(-1)
    cls=np.asarray(masks[t]); person=ndi.binary_erosion(np.isin(cls,[1,2,3]),iterations=4).astype(np.uint8)
    m=cv2.remap(person,uv[...,0]/1.5,uv[...,1]/1.5,cv2.INTER_NEAREST)>0; hair=cv2.remap((cls==1).astype(np.uint8),uv[...,0]/1.5,uv[...,1]/1.5,cv2.INTER_NEAREST)>0
    w=np.clip((fa-0.2)/0.8,0,1)**2*vis*m; return col,w.astype(np.float32),hair
gray=lambda x: cv2.cvtColor(np.clip(x,0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
nc=lambda x,w,s: cv2.GaussianBlur(x*(w[...,None] if x.ndim==3 else w),(0,0),s)/np.maximum(cv2.GaussianBlur(w,(0,0),s)[...,None] if x.ndim==3 else cv2.GaussianBlur(w,(0,0),s),1e-4)
# pull-push fill of what no key frame saw
def fill(img,m):
    lv=[(img*m[...,None],m.astype(np.float32))]
    while min(lv[-1][1].shape)>4: a,b=lv[-1]; lv.append((cv2.pyrDown(a).reshape(a.shape[0]//2 if a.shape[0]%2==0 else (a.shape[0]+1)//2,-1,a.shape[2]),cv2.pyrDown(b)))
    acc=lv[-1][0]/np.maximum(lv[-1][1],1e-4)[...,None]
    for a,b in lv[-2::-1]:
        up=cv2.resize(acc,(b.shape[1],b.shape[0]),interpolation=cv2.INTER_LINEAR).reshape(b.shape[0],b.shape[1],-1); wgt=np.clip(b*8,0,1)[...,None]; acc=wgt*(a/np.maximum(b,1e-4)[...,None])+(1-wgt)*up
    return acc
T=np.zeros((TH,TW,3),np.float32); Hc=np.zeros((TH,TW,3),np.float32); La=np.zeros((TH,TW,3),np.float32); Lw=np.zeros((TH,TW),np.float32); Wc=np.zeros((TH,TW),np.float32); hv=np.zeros((TH,TW),np.float32); sv=np.zeros((TH,TW),np.float32); log=[]
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); gx,gy=np.meshgrid(np.arange(TW,dtype=np.float32),np.arange(TH,dtype=np.float32))
for name,t in views:
    col,w,hair=project(t); shift=0.0; gain=[1,1,1]
    if name=='from above': w=w*np.clip((E2-30)/12,0,1).astype(np.float32)      # he frowns while bowing: use it for the scalp only
    if name=='from below': w=w*np.clip((-22-E2)/10,0,1).astype(np.float32)
    ov=((Wc>0.15)&(w>0.15)).astype(np.float32)
    if ov.sum()>3000:
        # colour: match the new view to what is already painted where they overlap, and carry that correction outward
        near=(cv2.GaussianBlur(ov,(0,0),14)>0.25).astype(np.float32); g=fill(nc(T,ov,14)/np.maximum(nc(col,ov,14),1),near); g=np.clip(cv2.GaussianBlur(g,(0,0),10),0.5,2.0)
        col=col*g; gain=[round(float(np.median(g[...,c][ov>0])),2) for c in range(3)]
        fl=dis.calc(gray(T),gray(col),None)                       # where does each painted texel sit in the new view?
        fl=np.clip(nc(fl,ov,12),-14,14)*np.clip(cv2.GaussianBlur(ov,(0,0),25)*6,0,1)[...,None]*(0.0 if name.startswith('back') else 1.0)   # back poses are approximate: no warping there
        col=cv2.remap(col,gx+fl[...,0],gy+fl[...,1],cv2.INTER_LINEAR); w=cv2.remap(w,gx+fl[...,0],gy+fl[...,1],cv2.INTER_LINEAR); shift=float(np.abs(fl[ov>0]).mean())
    blk=np.clip(Wc/0.25,0,1); wp=w**3*(1-blk*blk*(3-2*blk))+1e-7*(w>0)     # earlier (better) views keep what they already own
    # two bands: broad colour is averaged over every view that sees a texel (evens out the lighting, hides seams);
    # fine detail comes from the single best view (stays sharp)
    vm=(w>0.02).astype(np.float32); low=fill(nc(col,vm,8),vm); high=(col-low)*vm[...,None]; wl=cv2.GaussianBlur(wp,(0,0),22)
    La+=low*wl[...,None]; Lw+=wl; Hc=(Hc*Wc[...,None]+high*wp[...,None])/np.maximum(Wc+wp,1e-6)[...,None]; Wc=Wc+wp
    T=La/np.maximum(Lw,1e-6)[...,None]+Hc; hv+=w*hair; sv+=w*(~hair)
    log.append(dict(view=name,frame=int(t),time=round(t/30,2),view_az=round(float(np.degrees(np.arctan2(vdir(t)[0],vdir(t)[2])))),calm_mm=round(calm(t)*10,1),speed_deg=round(float(speed(t)),2),gain=gain,register_texels=round(shift,2),new_texels=int(((w>0.15)&(Wc-wp<=0.05)).sum())))
    print(log[-1],flush=True)
seen=Wc>0.01
T=fill(T,seen.astype(np.float32))
# close the loop at the back: the padded strip exists twice (once continuing each side); cross-fade the two copies
rp=np.linspace(0,1,2*PAD,dtype=np.float32)[None,:,None]; bl=T[:,:2*PAD]*rp+T[:,TW0:TW0+2*PAD]*(1-rp); T[:,:2*PAD]=bl; T[:,TW0:TW0+2*PAD]=bl
_top=np.clip((E2-52)/8,0,1)
hairmap=cv2.GaussianBlur(fill((hv/np.maximum(hv+sv,1e-4))[...,None],((hv+sv)>0.05).astype(np.float32))[...,0],(0,0),3)
hairmap=np.maximum(hairmap,_top)
# white balance: warm room light -> neutral skin
skin=seen&(hairmap<0.2)&(A2>-60)&(A2<60)&(E2>-35)&(E2<25)
lin=(np.clip(T,0,255)/255)**2.2; med=np.median(lin[skin],0); target=np.array([0.125,0.185,0.305]); lin=np.clip(lin*(target/med),0,1); out=(lin**(1/2.2)*255)
out=out[:,PAD:-PAD]; hairmap=hairmap[:,PAD:-PAD]; seen=seen[:,PAD:-PAD]
# harmonising view after view round the head leaves the two sides at different exposures where they meet at the back.
# Hair is one colour and so is the nape: flatten the broad colour there (and in all hair), keep the detail.
def wblur(x,s):
    p=int(4*s); y=cv2.GaussianBlur(np.concatenate([x[:,-p:],x,x[:,:p]],1),(0,0),s); return y[:,p:-p]
A2c=A2[:,PAD:-PAD]; bw=np.clip((np.abs(A2c)-95)/20,0,1); hs=np.clip((hairmap-0.35)/0.3,0,1)
for msk,wf in ((hs,np.ones_like(hs)),(1-hs,bw)):
    m=msk.astype(np.float32); lowc=wblur(out*m[...,None],35)/np.maximum(wblur(m,35),1e-3)[...,None]; medc=np.median(out[(msk>0.6)&seen&(np.abs(E2[:,PAD:-PAD])<60)],0)
    ratio=np.clip(medc/np.maximum(lowc,4),0.4,2.5); k=(0.85*msk*wf)[...,None]; out=out*(1-k)+np.clip(out*ratio,0,255)*k
cv2.imwrite('texture.jpg',out.astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,90]); np.save('work/hairmap.npy',hairmap.astype(np.float32)); np.save('work/seen.npy',seen)
r0=r0[:,PAD:-PAD]; E2=E2[:,PAD:-PAD]
json.dump(dict(wb=[float(x) for x in (target/med)],views=log,seen=float(seen[(r0>1.2)&(E2>-60)].mean()),front=int(views[0][1])),open('work/texture.json','w'),indent=1)
print('texture: %.0f%% of the head surface seen by a key frame'%(100*seen[(r0>1.2)&(E2>-60)].mean()))
cv2.imwrite('work/texture_prev.jpg',cv2.resize(out[140:860,424:1624],(900,540)).astype(np.uint8))
