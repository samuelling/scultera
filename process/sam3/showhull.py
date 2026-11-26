import numpy as np,sys; sys.path.insert(0,'proc')
from skimage import measure; from scipy import ndimage as ndi
from prev import sheet
h=np.load(sys.argv[1]); occ=ndi.gaussian_filter(h['occ'].astype(np.float32),1.0); xs,ys,zs=h['xs'],h['ys'],h['zs']; g=xs[1]-xs[0]
v,f,_,_=measure.marching_cubes(occ,0.5,spacing=(g,g,g)); v+=[xs[0],ys[0],zs[0]]
S=np.load('shape_s.npz')['S']
pts=[]
sheet([(v,f[:,::-1],(0.8,0.7,0.6))],sys.argv[2],views=((0,0),(90,0),(180,0),(-90,0),(0,60)),size=300,span=34,center=(0,-4,-5))
