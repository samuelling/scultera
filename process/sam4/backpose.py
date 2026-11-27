"""Take 4, step 1: a pose for every one of the 1,421 frames.
Tracked frames: PnP (take 3). Gaps (back of the head): chair yaw integrated from dense optical flow on the hair,
closed on the tracked poses at both ends, then refined against the head's silhouette."""
import numpy as np, cv2, sys, json
sys.path.insert(0,'proc'); from common import *
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as Rot, Slerp
good=GOOD; idx=sh['idx']; Rg=sh['R'][good]; tg=sh['tr'][good]; fg=idx[good]
masks=np.load(B3+'masks.npy',mmap_mode='r'); N,H,W=masks.shape
fin=np.load(B3+'finished.npz'); V=fin['v'].astype(float); rng=np.random.default_rng(0)
Pm=toold(V[V[:,1]>-7.0]); Pm=Pm[rng.choice(len(Pm),30000,replace=False)]; c_head=Pm.mean(0)
gray=lambda t: cv2.imread(B3+'frames/f%04d.jpg'%(t+1),0)
hmask=lambda t: np.isin(np.asarray(masks[t]),[1,3])
pos={fr:i for i,fr in enumerate(fg)}; gset=set(fg.tolist())
# chair axis (world vertical in camera coords) from the first tracked turn
def axis(a,b):
    rv=Rot.from_matrix(Rg[pos[b]]@Rg[pos[a]].T).as_rotvec(); return rv/np.linalg.norm(rv),np.degrees(np.linalg.norm(rv))
fa=[f_ for f_ in fg if 12<=f_<=20][0]; fb=[f_ for f_ in fg if 90<=f_<=104][-1]; n,ang=axis(fa,fb)
fa2=[f_ for f_ in fg if 555<=f_<=565][0]; fb2=[f_ for f_ in fg if 585<=f_<=600][-1]; n2,ang2=axis(fa2,fb2)
if n2@n<0: n2=-n2
print('chair axis from turn 1 (%.0f deg):'%ang,n.round(3),' turn 3 (%.0f deg):'%ang2,n2.round(3),' they differ by %.1f deg'%np.degrees(np.arccos(np.clip(n@n2,-1,1))))
# turn 3 includes a nod, so the chair axis is taken from turn 1 alone
def twist(Rrel):
    q=Rot.from_matrix(Rrel).as_quat(); return 2*np.degrees(np.arctan2(q[:3]@n,q[3]))
Rn=lambda deg: Rot.from_rotvec(n*np.radians(deg)).as_matrix()
def splat(Rm,t3,ds=2):
    X=Pm@Rm.T+t3; u=np.round((F0*X[:,0]/X[:,2]+CX)/ds).astype(int); v=np.round((F0*X[:,1]/X[:,2]+CY)/ds).astype(int)
    ok=(u>=1)&(u<W//ds-1)&(v>=1)&(v<H//ds-1); zb=np.full((H//ds,W//ds),np.inf)
    for du in (-1,0,1):
        for dv in (-1,0,1): np.minimum.at(zb,(v[ok]+dv,u[ok]+du),X[ok,2])      # nearest surface wins
    return zb,(v[ok].max() if ok.any() else 0)
def place(Rm,mk,Z):
    yy,xx=np.nonzero(mk)
    if len(xx)<50: return None
    # top-of-head anchored: centroid of the mask rows above the model's cut
    return np.array([(xx.mean()-CX)*Z/F0,(yy.mean()-CY)*Z/F0,Z])-Rm@c_head
def upper(mk,Rm,t3):
    X=Pm@Rm.T+t3; vmax=int((F0*X[:,1]/X[:,2]+CY).max()); m=mk.copy(); m[vmax:]=False; return m
def iou(Rm,t3,mk):
    zb,_=splat(Rm,t3,4); s=ndi.binary_closing(np.isfinite(zb),iterations=2); m=mk[::4,::4]
    return (s&m).sum()/max((s|m).sum(),1)
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gaps=[];t=0
while t<N:
    if t in gset: t+=1; continue
    s=t
    while t<N and t not in gset: t+=1
    if s>0 and t<N: gaps.append((s,t))
outR={fr:Rg[pos[fr]] for fr in fg}; outT={fr:tg[pos[fr]] for fr in fg}; src={fr:'tracked' for fr in fg}; report=[]
raw={}
for a,b in gaps:
    R0=outR[a-1]; t0=outT[a-1]; Z0=t0[2]; Z1=outT[b][2]; nfr=b-a
    th=[0.0]; Rm=R0; t3=t0.copy(); prev=gray(a-1)
    for k in range(nfr+1):
        t=a+k; cur=gray(t); fl=dis.calc(prev,cur,None)
        mk=ndi.binary_erosion(hmask(t-1),iterations=10); zb,_=splat(Rm,t3,2); zb=np.repeat(np.repeat(zb,2,0),2,1)
        sel=mk&np.isfinite(zb)
        if sel.sum()>400:
            vv,uu=np.nonzero(sel); Z=zb[vv,uu]; X=np.stack([(uu-CX)*Z/F0,(vv-CY)*Z/F0,Z],1); c=Rm@c_head+t3
            a_=F0/Z*np.cross(n,X-c)[:,0]                          # x-flow per radian of chair turn
            fx=fl[vv,uu,0]; A=np.c_[np.ones(len(a_)),a_]; w=np.ones(len(a_))
            for it in range(4):
                co=np.linalg.lstsq(A*w[:,None],fx*w,rcond=None)[0]; r=fx-A@co; w=1/np.maximum(1,np.abs(r)/0.7)
            om=np.degrees(co[1])
        else: om=0.0
        th.append(th[-1]+om); Rm=Rn(th[-1])@R0
        Z=Z0+(Z1-Z0)*(k+1)/(nfr+1); p=place(Rm,upper(hmask(t),Rm,t3),Z); t3=p if p is not None else t3; prev=cur
    th=np.array(th[1:]); true=twist(outR[b]@R0.T); cand=true+360*np.arange(-2,3); true=cand[np.argmin(np.abs(cand-th[-1]))]
    raw[(a,b)]=(th,true)
# one scale for the flow->angle conversion, chosen by silhouette agreement over the two long chair turns
# anchors: the instants he is seen from exactly behind = both ears show equal skin either side of the hair's centre line
anchors={}; y_of=lambda t: twist(outR[t]@outR[0].T)
for a,b in gaps:
    if b-a<100: continue
    th,_=raw[(a,b)]; asym=[]
    for t in range(a,b):
        m=np.asarray(masks[t]); hair=m==1; xs=np.nonzero(m==3)[1]
        asym.append(0 if hair.sum()<500 or len(xs)<50 else ((xs<np.nonzero(hair)[1].mean()).sum()*2-len(xs))/len(xs))
    asym=ndi.gaussian_filter1d(np.array(asym,float),3); y0=y_of(a-1); pk=th[np.argmax(np.abs(th))]
    target=np.sign(pk)*180-y0; z=np.where(np.diff(np.sign(asym))!=0)[0]
    anchors[(a,b)]=[(int(i),float(target)) for i in z if abs(1.2*th[i]-target)<45]
    print('gap %.1f-%.1fs: seen from exactly behind at'%(a/30,b/30),[round((a+i)/30,2) for i,_ in anchors[(a,b)]],'-> turn there must be %.0f deg; flow alone says'%target,[round(float(th[i])) for i,_ in anchors[(a,b)]])
def path(a,b,k):
    th,true=raw[(a,b)]; nfr=b-a; kt=[-1.0]; kc=[0.0]
    for i,target in anchors.get((a,b),[]): kt.append(i); kc.append(target-k*th[i])
    kt.append(nfr); kc.append(true-k*th[nfr]); return k*th+np.interp(np.arange(nfr+1),kt,kc), true
def poses(a,b,ths,true,every=1):
    R0=outR[a-1]; Qb=Rn(-true)@outR[b]; sl=Slerp([0,1],Rot.from_matrix(np.stack([R0,Qb]))); nfr=b-a; out=[]
    for k in range(0,nfr,every):
        Rm=Rn(ths[k])@sl((k+1)/(nfr+1)).as_matrix(); mk=hmask(a+k); Z=outT[a-1][2]+(outT[b][2]-outT[a-1][2])*(k+1)/(nfr+1)
        t3=place(Rm,mk,Z)
        if t3 is None: out.append((a+k,Rm,None,mk)); continue
        t3=place(Rm,upper(mk,Rm,t3),Z); out.append((a+k,Rm,t3,mk))
    return out
long=[g for g in gaps if g[1]-g[0]>100][:2]; tab={}
for k in (0.7,0.85,1.0,1.15,1.3,1.5):
    sc=[]
    for a,b in long:
        ths,true=path(a,b,k); sc+= [iou(Rm,t3,upper(mk,Rm,t3)) for _,Rm,t3,mk in poses(a,b,ths,true,6) if t3 is not None]
    tab[k]=float(np.mean(sc)); print('flow scale x%.2f: mean silhouette IoU %.3f  (peak turn in gap 1: %.0f deg)'%(k,tab[k],np.abs(path(*long[0],k)[0]).max()))
kbest=1.15   # silhouettes can't tell the scales apart (table above is flat); the ear anchors imply 1.13-1.3
DEL=np.arange(-30,31,3.0)
for a,b in gaps:
    ths,true=path(a,b,kbest); nfr=b-a
    if nfr>6:   # silhouette refinement around the flow path (dynamic programming, smooth corrections)
        Rb=Slerp([0,1],Rot.from_matrix(np.stack([outR[a-1],Rn(-true)@outR[b]])))
        cost=np.ones((nfr,len(DEL)))
        for k in range(nfr):
            mk=hmask(a+k); Z=outT[a-1][2]+(outT[b][2]-outT[a-1][2])*(k+1)/(nfr+1); Rk=Rb((k+1)/(nfr+1)).as_matrix()
            for j,d in enumerate(DEL):
                Rm=Rn(ths[k]+d)@Rk; t3=place(Rm,mk,Z)
                if t3 is None: continue
                cost[k,j]=1-iou(Rm,t3,upper(mk,Rm,t3))
        lam=0.02; prior=2e-4*DEL**2; tr_=lam*((DEL[:,None]-DEL[None,:])/3)**2
        acc=cost[0]+prior+lam*(DEL/3)**2; bk=np.zeros((nfr,len(DEL)),int)
        for k in range(1,nfr): m=acc[:,None]+tr_; bk[k]=m.argmin(0); acc=m.min(0)+cost[k]+prior
        j=int(np.argmin(acc+lam*(DEL/3)**2)); pth=[j]
        for k in range(nfr-1,0,-1): j=bk[k,j]; pth.append(j)
        d=ndi.gaussian_filter1d(DEL[pth[::-1]],2.0); ths2=ths[:nfr]+d
    else: ths2=ths[:nfr]; d=np.zeros(nfr)
    ious=[]
    for (t,Rm,t3,mk) in poses(a,b,ths2,true):
        if t3 is None: t3=outT[a-1]
        outR[t]=Rm; outT[t]=t3; src[t]='solved'; ious.append(iou(Rm,t3,upper(mk,Rm,t3)))
    if nfr>6:
        report.append(dict(t0=round(a/30,2),t1=round(b/30,2),frames=int(nfr),peak_turn=round(float(ths2[np.argmax(np.abs(ths2))]),0),loop_gap_deg=round(float(true-raw[(a,b)][0][-1]*kbest),1),sil_fix_deg=round(float(np.abs(d).max()),0),iou=round(float(np.mean(ious)),3)))
        print(report[-1])
frs=np.arange(N); np.savez('poses4.npz',R=np.array([outR[k] for k in frs]),tr=np.array([outT[k] for k in frs]),src=np.array([src[k] for k in frs]),n=n,kbest=kbest)
json.dump(dict(scale_table=tab,kbest=kbest,gaps=report,anchors={'%.2f-%.2f'%(a/30,b/30):[[round((a+i)/30,2),round(tg_)] for i,tg_ in v] for (a,b),v in anchors.items()},raw={'%d-%d'%k_:[round(float(x),1) for x in v[0]] for k_,v in raw.items() if k_[1]-k_[0]>100}),open('work/backpose.json','w'),indent=1)
