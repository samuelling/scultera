import numpy as np
from collections import defaultdict
def edges_of(F):
    E=np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),1)
    u,inv,cnt=np.unique(E,axis=0,return_inverse=True,return_counts=True)
    return u,inv.reshape(3,-1).T,cnt
def boundary_loops(F):
    E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); s=set(map(tuple,E))
    bd=[(a,b) for a,b in E if (b,a) not in s]; nxt={a:b for a,b in bd}; loops=[];seen=set()
    for a,_ in bd:
        if a in seen: continue
        L=[a];seen.add(a);c=nxt[a]
        while c!=a: L.append(c);seen.add(c);c=nxt[c]
        loops.append(L)
    return loops
def loop_subdivide(V,F,attrs=()):
    """one level of Loop subdivision (boundary-aware). attrs: per-vertex arrays interpolated linearly at edge midpoints, kept at verts."""
    E,FE,cnt=edges_of(F); nV=len(V)
    bnd=cnt==1
    # edge points
    opp=defaultdict(list)
    for f in F:
        for i in range(3):
            a,b,c=f[i],f[(i+1)%3],f[(i+2)%3]; opp[(min(a,b),max(a,b))].append(c)
    EP=np.zeros((len(E),3))
    for k,(a,b) in enumerate(E):
        o=opp[(a,b)]
        EP[k]=(V[a]+V[b])/2 if (bnd[k] or len(o)<2) else 3/8*(V[a]+V[b])+1/8*(V[o[0]]+V[o[1]])
    # vertex points
    nb=[[] for _ in range(nV)]; bnb=[[] for _ in range(nV)]
    for k,(a,b) in enumerate(E):
        nb[a].append(b);nb[b].append(a)
        if bnd[k]: bnb[a].append(b);bnb[b].append(a)
    VP=V.copy()
    for i in range(nV):
        if bnb[i]:
            if len(bnb[i])==2: VP[i]=3/4*V[i]+1/8*(V[bnb[i][0]]+V[bnb[i][1]])
        elif nb[i]:
            n=len(nb[i]); beta=(5/8-(3/8+np.cos(2*np.pi/n)/4)**2)/n
            VP[i]=(1-n*beta)*V[i]+beta*V[nb[i]].sum(0)
    NV=np.r_[VP,EP]; e=FE+nV
    NF=np.concatenate([np.c_[F[:,0],e[:,0],e[:,2]],np.c_[F[:,1],e[:,1],e[:,0]],np.c_[F[:,2],e[:,2],e[:,1]],e])
    out=[]
    for A in attrs:
        out.append(np.r_[A,(A[E[:,0]]+A[E[:,1]])/2])
    return NV,NF,out
