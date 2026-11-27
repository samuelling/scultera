"""Take 4, step 3 (an experiment): can frame pairs ~14 deg apart measure the face's relief by plane-sweep stereo?
Two disjoint sets of pairs are solved independently; if their relief maps agree, it is signal. If not, it is not used."""
import numpy as np, cv2, sys, json
sys.path.insert(0,'proc'); from common import *
import sdf4
from scipy import ndimage as ndi
p=np.load('poses4.npz'); R,tr,src=p['R'],p['tr'],p['src']; C=sdf4.C
az=np.arange(-72,72,0.25)+0.125; el=np.arange(-50,44,0.25)+0.125; D=sdf4.dirs(az,el); hgt,wid=D.shape[:2]
r0=sdf4.trace(D,parts=('head','face','neck')); P0=C+r0[...,None]*D
nrm=np.cross(np.gradient(P0,axis=1),np.gradient(P0,axis=0)); nrm/=np.linalg.norm(nrm,axis=-1,keepdims=True); nrm*=np.sign((nrm*D).sum(-1,keepdims=True))
A2,E2=np.meshgrid(az,el); face=(sdf4.fmap(sdf4.WF,A2,E2)>0.3)&(sdf4.fmap(sdf4.Lab.astype(np.float32),A2,E2,0)==0)
# frames: tracked, calm face
good=sh['good']; idx=sh['idx']; Dx=fc['D']; calm={}
MOV=[13,14,17,0,61,291,152,159,145,386,374,105,334]
for k,t in enumerate(idx):
    if good[k]: calm[int(t)]=float(np.abs(Dx[k,MOV]).mean())
thr=np.percentile(list(calm.values()),65); ok=[t for t in calm if calm[t]<thr]
cam=lambda t: tonew((-R[t].T@tr[t])[None])[0]
vdir=lambda t: (cam(t)-C)/np.linalg.norm(cam(t)-C)
vaz=lambda t: np.degrees(np.arctan2(vdir(t)[0],vdir(t)[2]))
pairs=[]; used=set()
for tgt in list(range(-58,60,4))+list(range(-56,60,4)):
    free=[t for t in ok if all(abs(t-u)>6 for u in used)]
    if not free: continue
    a=min(free,key=lambda t: abs(vaz(t)-tgt)+0.3*abs(np.degrees(np.arcsin(vdir(t)[1]))-15)); best=None
    if abs(vaz(a)-tgt)>8: continue
    for b in free:
        if abs(b-a)>45 or abs(b-a)<7: continue
        ang=np.degrees(np.arccos(np.clip(vdir(a)@vdir(b),-1,1)))
        if 11<ang<18 and (best is None or abs(ang-14)<best[1]): best=(b,abs(ang-14),ang)
    if best: pairs.append((a,best[0],round(float(vaz(a))),round(float(best[2]),1))); used|={a,best[0]}
print('pairs (frame A, frame B, view azimuth, baseline deg):',pairs)
def img(t): return cv2.GaussianBlur(cv2.imread('full/f%04d.jpg'%(t+1),0).astype(np.float32),(0,0),0.8)
def sample(I,t,P):
    uv,z=proj(toold(P.reshape(-1,3)),R[t],tr[t],1.5); m=uv.reshape(hgt,wid,2).astype(np.float32); return cv2.remap(I,m[...,0],m[...,1],cv2.INTER_LINEAR)
def vis(t): v=cam(t)-P0; v/=np.linalg.norm(v,axis=-1,keepdims=True); return (nrm*v).sum(-1)
DEL=np.arange(-0.6,0.601,0.05); box=lambda x: cv2.blur(x,(17,17))
def solve(a,b):
    Ia,Ib=img(a),img(b); m=(vis(a)>0.35)&(vis(b)>0.35)&face
    A0=sample(Ia,a,P0); B0=sample(Ib,b,P0)
    try:   # take out the mean misregistration (pose error) before sweeping
        W_=np.eye(2,3,dtype=np.float32); _,W_=cv2.findTransformECC(A0,B0,W_,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-4),m.astype(np.uint8),5)
    except cv2.error: W_=np.eye(2,3,dtype=np.float32)
    vol=np.empty((len(DEL),hgt,wid),np.float32)
    for k,d in enumerate(DEL):
        P=P0+d*D; A=sample(Ia,a,P); B=cv2.warpAffine(sample(Ib,b,P),W_,(wid,hgt),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        ma,mb=box(A),box(B); va=box(A*A)-ma*ma; vb=box(B*B)-mb*mb; vol[k]=cv2.GaussianBlur((box(A*B)-ma*mb)/np.sqrt(np.maximum(va*vb,1e-3))*(va>2.0),(0,0),1.5)
    k=vol.argmax(0); pk=vol.max(0); kk=np.clip(k,1,len(DEL)-2); I,J=np.indices(k.shape)
    y0,y1,y2=vol[kk-1,I,J],vol[kk,I,J],vol[kk+1,I,J]; sub=np.clip(0.5*(y0-y2)/np.minimum(y0-2*y1+y2,-1e-6),-1,1)
    d=DEL[kk]+sub*0.05; conf=np.where(m&(k>0)&(k<len(DEL)-1)&(pk>0.45),pk-np.median(vol,0),0)
    return d,conf,float(np.hypot(*W_[:,2]))
def highpass(d,w,s=20):
    lo=cv2.GaussianBlur(d*w,(0,0),s)/np.maximum(cv2.GaussianBlur(w,(0,0),s),1e-3); return d-lo
sets=[[],[]]; info=[]
for i,(a,b,va_,bl) in enumerate(pairs):
    d,cf,shift=solve(a,b); sets[i%2].append((d,cf)); info.append(dict(a=int(a),b=int(b),view_az=va_,baseline=bl,shift_texels=round(shift,2),covered=round(float((cf>0).mean()),3),peak=round(float(np.median(cf[cf>0])) if (cf>0).any() else 0,2)))
    print(info[-1])
out=[]
for S_ in sets:
    w=sum(c for _,c in S_); d=sum(d*c for d,c in S_)/np.maximum(w,1e-6); hp=highpass(d,(w>0).astype(np.float32)*np.minimum(w,1)); hp=cv2.GaussianBlur(hp*(w>0),(0,0),5)/np.maximum(cv2.GaussianBlur((w>0).astype(np.float32),(0,0),5),1e-3); out.append((hp,w))
both=(out[0][1]>0.05)&(out[1][1]>0.05)&face
r=float(np.corrcoef(out[0][0][both],out[1][0][both])[0,1]); rms=[float(np.sqrt((o[0][both]**2).mean())*10) for o in out]
print('held-out agreement between the two independent sets: r = %.2f over %d texels; relief rms %.2f / %.2f mm'%(r,both.sum(),rms[0],rms[1]))
np.savez('work/stereoB.npz',hp0=out[0][0],hp1=out[1][0],w0=out[0][1],w1=out[1][1],az=az,el=el,face=face,r=r)
json.dump(dict(pairs=info,r=r,rms_mm=rms,texels=int(both.sum())),open('work/stereoB.json','w'),indent=1)
def heat(x,m,s=0.15):
    c=np.clip(x/s*0.5+0.5,0,1); im=cv2.applyColorMap((c*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS); im[~m]=(40,36,33); return im[::-1]
cv2.imwrite('work/stereoB.png',np.hstack([heat(out[0][0],(out[0][1]>0.05)&face),heat(out[1][0],(out[1][1]>0.05)&face)]))
