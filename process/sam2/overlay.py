"""Project the sculpt into video frames with the tracked pose: compare its silhouette to the segmentation."""
import numpy as np, sys
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
m=np.load(sys.argv[1]); v=m['v']; fc=np.load('face.npz'); d=np.load('shape_s.npz')
R,tr,f,cx,cy=d['R'],d['tr'],float(d['f']),float(d['cx']),float(d['cy'])
masks=np.load('masks.npy')
vo=(v+np.array([fc['shx'],0,0])-fc['m0'])@fc['Rh']+fc['m0']
ims=[];stats=[]
for t in [10,77,150,230,300]:
    X=vo@R[t].T+tr[t]; u=f*X[:,0]/X[:,2]+cx; w=f*X[:,1]/X[:,2]+cy
    sil=np.zeros(masks[t].shape,bool); ok=(u>=0)&(u<sil.shape[1])&(w>=0)&(w<sil.shape[0]); sil[w[ok].astype(int),u[ok].astype(int)]=True
    sil=ndi.binary_closing(sil,iterations=3); sil=ndi.binary_fill_holes(sil)
    head=np.isin(masks[t],[1,3])
    im=np.array(Image.open('frames/f%04d.jpg'%(t+1))).astype(float)*0.55
    e1=sil^ndi.binary_erosion(sil,iterations=2); e2=head^ndi.binary_erosion(head,iterations=2)
    im[e2]=[255,80,80]; im[e1]=[80,255,120]
    ims.append(Image.fromarray(im.astype(np.uint8)).crop((60,150,660,900)).resize((300,375)))
    # width comparison at a few rows
    rows=[int(np.percentile(np.where(head.any(1))[0],p)) for p in (15,40,60)]
    stats.append([(int(head[r].sum()),int(sil[r].sum())) for r in rows])
c=Image.new('RGB',(1500,375));[c.paste(im,(300*i,0)) for i,im in enumerate(ims)];c.save(sys.argv[2])
print('row widths (video head px, sculpt px):',stats)
