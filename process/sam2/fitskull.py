"""Fit the skull ellipsoid to the video silhouettes (rows above the eyes): left/right/top extents."""
import numpy as np
from scipy.optimize import minimize
fc=np.load('face.npz'); d=np.load('shape_s.npz'); S=fc['S']
R,tr,f,cx,cy=d['R'],d['tr'],float(d['f']),float(d['cx']),float(d['cy'])
masks=np.load('masks.npy')
T=list(range(0,358,12))
th,ph=np.meshgrid(np.linspace(0,np.pi,60),np.linspace(-np.pi,np.pi,120))
U=np.stack([np.sin(th)*np.sin(ph),np.cos(th),np.sin(th)*np.cos(ph)],-1).reshape(-1,3)
def toold(P): return (P+np.array([fc['shx'],0,0])-fc['m0'])@fc['Rh']+fc['m0']
targets=[]
for t in T:
    hd=np.isin(masks[t],[1,3]); rows=np.where(hd.any(1))[0]; top=rows.min()
    X=toold(S)@R[t].T+tr[t]; ey=np.median(f*X[[159,386],1]/X[[159,386],2]+cy)
    rr=np.linspace(top+4,ey-10,12).astype(int)
    ext=[(np.where(hd[r])[0].min(),np.where(hd[r])[0].max()) for r in rr]
    targets.append((t,top,rr,np.array(ext,float)))
def yawm(a):
    a=np.radians(a); return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def loss(p,verbose=False):
    rx,ry,rz,yc,zc,xc,yw=p; c=np.array([xc,yc,zc]); P=(U*[rx,ry,rz])@yawm(yw).T+c
    P=np.r_[P,S[:468]]
    L=0;n=0
    for t,top,rr,ext in targets:
        X=toold(P)@R[t].T+tr[t]; u=f*X[:,0]/X[:,2]+cx; v=f*X[:,1]/X[:,2]+cy
        L+=(v.min()-top)**2; n+=1
        for r,(a,b) in zip(rr,ext):
            sel=np.abs(v-r)<3
            if sel.sum()<3: L+=400; continue
            L+=(u[sel].min()-a)**2+(u[sel].max()-b)**2; n+=2
    return L/n
p0=np.array([8.2,8.8,9.45,2.4,-8.9,0,0])
print('start rms px %.1f'%np.sqrt(loss(p0)))
res=minimize(loss,p0,method='Powell',options=dict(maxiter=4000,xtol=0.01,ftol=1e-4))
print('fit rms px %.1f'%np.sqrt(res.fun),'params rx ry rz yc zc xc yaw =',res.x.round(2))
np.save('skull_fit.npy',res.x)
p=res.x; c=np.array([p[5],p[3],p[4]]); P=np.r_[(U*p[:3])@yawm(p[6]).T+c,S[:468]]
lo=[];hi=[];tp=[]
for t,top,rr,ext in targets:
    X=toold(P)@R[t].T+tr[t]; u=f*X[:,0]/X[:,2]+cx; v=f*X[:,1]/X[:,2]+cy; tp.append(v.min()-top)
    for r,(a,b) in zip(rr,ext):
        sel=np.abs(v-r)<3
        if sel.sum()>=3: lo.append(u[sel].min()-a); hi.append(u[sel].max()-b)
print('left edge err mean %.1f sd %.1f | right edge err mean %.1f sd %.1f | top %.1f'%(np.mean(lo),np.std(lo),np.mean(hi),np.std(hi),np.mean(tp)))
