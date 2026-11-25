import sys,numpy as np
from PIL import Image
im=np.asarray(Image.open(sys.argv[1]).convert('RGBA')).astype(np.float32)/255
h,w=im.shape[:2];yy,xx=np.mgrid[0:h,0:w]
r=np.sqrt(((xx-w*0.45)/w)**2+((yy-h*0.55)/h)**2)
top=np.array([0.80,0.78,0.75]);edge=np.array([0.52,0.50,0.48])
bg=top+(edge-top)*np.clip(r/0.85,0,1)[...,None]
a=im[...,3:4];out=im[...,:3]*a+bg*(1-a)
Image.fromarray((np.clip(out,0,1)*255).astype(np.uint8)).save(sys.argv[2])
