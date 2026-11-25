import numpy as np, time
from scipy.interpolate import CubicSpline
from skimage.measure import marching_cubes
t0=time.time()
h=0.025
X0,X1,Y0,Y1,Z0,Z1=-5.7,5.9,-3.1,3.1,-0.12,2.3
xs=np.arange(X0,X1,h,dtype=np.float32);ys=np.arange(Y0,Y1,h,dtype=np.float32);zs=np.arange(Z0,Z1,h,dtype=np.float32)
F=np.full((len(xs),len(ys),len(zs)),1e3,np.float32)
def sl(lo,hi,arr,o):
    i0=max(0,int((lo-o)/h)-1);i1=min(len(arr),int((hi-o)/h)+2);return slice(i0,i1)
def sub(c,ext):
    sx=sl(c[0]-ext[0],c[0]+ext[0],xs,X0);sy=sl(c[1]-ext[1],c[1]+ext[1],ys,Y0);sz=sl(c[2]-ext[2],c[2]+ext[2],zs,Z0)
    return (sx,sy,sz),xs[sx][:,None,None]-c[0],ys[sy][None,:,None]-c[1],zs[sz][None,None,:]-c[2]
def smin(a,b,k):
    hh=np.clip(0.5+0.5*(b-a)/k,0,1);return b*(1-hh)+a*hh-k*hh*(1-hh)
def smax(a,b,k): return -smin(-a,-b,k)
PRIM=[]
def ellip(c,r,k):
    PRIM.append([round(float(v),3) for v in (*c,*r)])
    m=max(r)+k+0.1
    s,px,py,pz=sub(c,(m,m,m))
    k0=np.sqrt((px/r[0])**2+(py/r[1])**2+(pz/r[2])**2)
    k1=np.sqrt((px/r[0]**2)**2+(py/r[1]**2)**2+(pz/r[2]**2)**2)+1e-9
    d=k0*(k0-1)/k1
    F[s]=smin(d,F[s],k)
def sphere(c,r,k):
    ellip(c,(r,r,r),k)
# spine: x, y0, z0, W, H
sp=np.array([
[-5.00,0,0.32,0.20,0.18],
[-4.85,0,0.38,0.34,0.28],
[-4.60,0,0.50,0.56,0.40],
[-4.20,0,0.62,0.80,0.52],
[-3.70,0,0.72,0.98,0.62],
[-3.30,0,0.76,1.08,0.66],
[-2.90,0,0.78,1.02,0.64],
[-2.50,0,0.78,0.80,0.60],
[-2.10,0,0.80,0.92,0.66],
[-1.60,0,0.82,1.22,0.76],
[-0.80,0,0.82,1.40,0.80],
[ 0.00,0,0.80,1.42,0.79],
[ 0.80,0,0.78,1.30,0.76],
[ 1.50,0,0.70,1.05,0.66],
[ 2.00,0,0.60,0.95,0.58],
[ 2.80,-0.08,0.54,1.00,0.52],
[ 3.60,-0.30,0.46,0.86,0.45],
[ 4.40,-0.62,0.34,0.58,0.33],
[ 5.00,-0.92,0.24,0.32,0.23],
[ 5.35,-1.12,0.14,0.13,0.11]])
cs=[CubicSpline(sp[:,0],sp[:,j]) for j in range(1,5)]
for xv in np.arange(-4.98,5.38,0.07):
    y0,z0,W,Hh=[float(f(xv)) for f in cs]
    ellip((xv,y0,z0),(0.32,W,Hh),0.12)
def limb(pts,rads,k=0.08,step=0.05):
    pts=np.array(pts,float)
    for i in range(len(pts)-1):
        a,b=pts[i],pts[i+1];ra,rb=rads[i],rads[i+1]
        n=max(2,int(np.linalg.norm(b-a)/step))
        for t in np.linspace(0,1,n):
            sphere(tuple(a+(b-a)*t),ra+(rb-ra)*t,k)
def toes(w,dirs,L,r0=0.075):
    for ang,ln in zip(dirs,L):
        d=np.array([np.cos(ang),np.sin(ang),0]);
        tip=np.array(w)+d*ln;tip[2]=0.055
        mid=(np.array(w)+tip)/2;mid[2]=0.07
        limb([w,mid,tip],[r0*1.3,r0,r0*0.8],k=0.03,step=0.025)
        sphere(tuple(tip+d*0.02),r0*0.85,0.02)
for sgn in (-1,1):
    # front leg: shoulder -> elbow -> wrist (limp, reaching forward)
    sh=(-1.75,sgn*0.95,0.55);el=(-1.85,sgn*1.8,0.24);wr=(-2.55,sgn*2.15,0.13)
    limb([sh,el,wr],[0.26,0.18,0.13])
    base=np.pi if True else 0
    toes(wr,[np.pi+sgn*a for a in (-0.15,0.25,0.62,1.0,1.4)],[0.42,0.5,0.5,0.45,0.34])
    # hind leg: hip -> knee -> ankle (splayed back)
    hp=(1.25,sgn*0.95,0.50);kn=(1.55,sgn*2.0,0.28);an=(2.35,sgn*2.25,0.13)
    limb([hp,kn,an],[0.34,0.22,0.15])
    toes(an,[sgn*a for a in (-0.2,0.25,0.65,1.05,1.45)],[0.5,0.6,0.62,0.55,0.4])
# eyes: sockets + lids
r_eye=0.30
eyes=[]
for sgn in (-1,1):
    c=np.array([-3.60,sgn*0.76,0.98])
    o=np.array([-0.15,sgn*0.66,0.75]);o/=np.linalg.norm(o)
    up=np.array([0,0,1.])-o[2]*o;up/=np.linalg.norm(up)
    bk=np.array([1.,0,0]);bk=bk-bk.dot(o)*o-bk.dot(up)*up;bk/=np.linalg.norm(bk)
    eyes.append((c,o,up,bk))
    m=0.7
    s,px,py,pz=sub(c,(m,m,m))
    dist=np.sqrt(px**2+py**2+pz**2)
    F[s]=smax(F[s],-(dist-(r_eye+0.012)),0.03)
    dot=lambda v: px*v[0]+py*v[1]+pz*v[2]
    shell=np.abs(dist-0.330)-0.040
    nl=up+0.5*bk;nl/=np.linalg.norm(nl)
    upper=np.maximum(np.maximum(shell,(-0.105)-dot(nl)),-0.02-dot(o))
    lower=np.maximum(np.maximum(shell,0.19-dot(-up)),-0.02-dot(o))
    lids=np.minimum(upper,lower)
    F[s]=smin(lids,F[s],0.045)
print('sdf',time.time()-t0)
v,f,_,_=marching_cubes(F,0.0,spacing=(h,h,h))
v=v+np.array([X0,Y0,Z0])
# orient faces outward
fn=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
cen=v[f].mean(1)
if (fn*(cen-v.mean(0))).sum(1).mean()<0: f=f[:,[0,2,1]]
import json;json.dump(PRIM,open('prims.json','w'));np.savez('base.npz',v=v.astype(np.float32),f=f.astype(np.int32),eyes=np.array([[*e[0],*e[1],*e[2],*e[3]] for e in eyes]))
print(len(v),len(f),time.time()-t0)
