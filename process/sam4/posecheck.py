import numpy as np, cv2, sys
sys.path.insert(0,'proc'); from common import *
from scipy.spatial.transform import Rotation as Rot
p=np.load('poses4.npz'); R,tr,n=p['R'],p['tr'],p['n']; fin=np.load(B3+'finished.npz'); V=fin['v'].astype(float); F=fin['f']
keep=V[:,1]>-12; Po=toold(V); tiles=[]
q=lambda Rr: Rot.from_matrix(Rr).as_quat()
yaw=np.degrees(np.unwrap(np.radians([2*np.degrees(np.arctan2(q(R[t]@R[0].T)[:3]@n,q(R[t]@R[0].T)[3])) for t in range(len(R))])))
np.save('work/yaw.npy',yaw)
for t in [130,160,189,213,230,260,450,480,1241,1281]:
    im=cv2.imread(B3+'frames/f%04d.jpg'%(t+1)); uv,z=proj(Po[keep],R[t],tr[t]); sil=np.zeros(im.shape[:2],np.uint8)
    u=np.round(uv[:,0]).astype(int); v=np.round(uv[:,1]).astype(int); ok=(u>1)&(u<718)&(v>1)&(v<1278); sil[v[ok],u[ok]]=255
    sil=cv2.morphologyEx(sil,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8)); cs,_=cv2.findContours(sil,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(80,255,120),2)
    c=uv.mean(0).astype(int); cr=im[max(c[1]-230,0):c[1]+250,max(c[0]-200,0):c[0]+200]; cr=cv2.resize(cr,(200,240)); cv2.putText(cr,'%.1fs yaw %d'%(t/30,yaw[t]),(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1); tiles.append(cr)
cv2.imwrite('work/posecheck.jpg',np.hstack(tiles))
