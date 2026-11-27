import numpy as np
B3='/home/claude/sam3/'
sh=np.load(B3+'shape_s.npz'); F0=float(sh['f']); CX=float(sh['cx']); CY=float(sh['cy'])
fc=np.load(B3+'face.npz'); _m0=fc['m0']; _Rh=fc['Rh']; _sx=np.array([float(fc['shx']),0,0])
def toold(P): return (P+_sx-_m0)@_Rh+_m0          # sculpt frame -> pose frame
def tonew(P): return (P-_m0)@_Rh.T+_m0-_sx
def proj(P_old,R,t,s=1.0):
    X=P_old@R.T+t; return np.stack([s*(F0*X[...,0]/X[...,2]+CX),s*(F0*X[...,1]/X[...,2]+CY)],-1),X[...,2]
# frames the outlier test threw out for a jaw 3-4 cm open are not outliers: they are the yawn (26.5-29 s)
_m=np.abs(sh['D']).max((1,2)); GOOD=sh['good']|((sh['idx']>=796)&(sh['idx']<872)&(sh['err']<15)&(_m<4.5))
