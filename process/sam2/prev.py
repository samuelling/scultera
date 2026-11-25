"""Fast software preview: rasterize triangles with numpy z-buffer (no Blender), Lambert + rim."""
import numpy as np
from PIL import Image
def vnormals(V,F):
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]); N=np.zeros_like(V)
    for i in range(3): np.add.at(N,F[:,i],n)
    return N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)
def rot(yaw,pitch):
    a,b=np.radians(yaw),np.radians(pitch)
    Ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]);Rx=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    return Rx@Ry
def render(meshes,yaw=0,pitch=0,size=480,span=34,center=(0,-6,-6),light=(-0.5,0.6,0.8)):
    """meshes: list of (V,F,color). orthographic, camera looks along -z after rotating the model by yaw/pitch"""
    Rm=rot(yaw,pitch); L=np.array(light,float);L/=np.linalg.norm(L)
    Z=np.full((size,size),-1e9);C=np.zeros((size,size,3))
    ss=size/span
    for V,F,col in meshes:
        Vr=(V-center)@Rm.T; N=vnormals(V,F)@Rm.T
        px=Vr[:,0]*ss+size/2; py=size/2-Vr[:,1]*ss
        sh=np.clip(N@L,0,1)*0.75+0.18+0.25*np.clip(1-N[:,2],0,1)**3
        # supersample triangles: barycentric sample points proportional to area
        ux=px[F[:,1]]-px[F[:,0]];uy=py[F[:,1]]-py[F[:,0]];vx=px[F[:,2]]-px[F[:,0]];vy=py[F[:,2]]-py[F[:,0]]; a=np.abs(ux*vy-uy*vx)/2
        k=np.clip(np.ceil(np.sqrt(a)*2.6),1,60).astype(int)
        for kk in np.unique(k):
            sel=np.where(k==kk)[0]; g=np.array([(i,j) for i in range(kk+1) for j in range(kk+1-i)],float)/kk
            b=np.c_[g,1-g.sum(1)]
            for (b0,b1,b2) in b:
                f=F[sel]; x=b0*px[f[:,0]]+b1*px[f[:,1]]+b2*px[f[:,2]]; y=b0*py[f[:,0]]+b1*py[f[:,1]]+b2*py[f[:,2]]
                z=b0*Vr[f[:,0],2]+b1*Vr[f[:,1],2]+b2*Vr[f[:,2],2]; s=b0*sh[f[:,0]]+b1*sh[f[:,1]]+b2*sh[f[:,2]]
                xi=np.round(x).astype(int);yi=np.round(y).astype(int);ok=(xi>=0)&(xi<size)&(yi>=0)&(yi<size)
                xi,yi,z,s=xi[ok],yi[ok],z[ok],s[ok]
                o=np.argsort(z); xi,yi,z,s=xi[o],yi[o],z[o],s[o]     # later writes (nearer) win
                upd=z>Z[yi,xi]; Z[yi[upd],xi[upd]]=z[upd]; C[yi[upd],xi[upd]]=s[upd,None]*np.array(col)
    bg=np.array([0.12,0.12,0.13]); C[Z<-1e8]=bg
    return Image.fromarray((np.clip(C,0,1)*255).astype(np.uint8))
def sheet(meshes,out,views=((0,0),(35,0),(90,0)),**kw):
    ims=[render(meshes,y,p,**kw) for y,p in views]; W=sum(i.width for i in ims)
    c=Image.new('RGB',(W,ims[0].height)); x=0
    for i in ims: c.paste(i,(x,0)); x+=i.width
    c.save(out)
