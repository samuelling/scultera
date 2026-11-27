"""glTF for take 4: positions, normals, 14 morph targets; COLOR_0 = paint (alpha says where the texture applies), _CLAY = clay tones."""
import numpy as np, json, struct, cv2, sys
sys.path.insert(0,'proc'); import sdf4
from scipy import sparse as sp, ndimage as ndi
fin=np.load('finished4.npz'); an=np.load('anim4.npz'); Wv=sp.load_npz('Wv4.npz'); K=int(an['K']); CM=0.01
V=fin['v'].astype(np.float64); F=fin['f']; mat=fin['mat']
tex=(cv2.imread('texture.jpg')[:,:,::-1]/255.0)**2.2; TH,TW=tex.shape[:2]; rr,az,el=sdf4.angles(V)
P=np.stack([ndi.map_coordinates(tex[...,c],[(90-el)/180*TH-0.5,(az+180)/360*TW-0.5],order=1,mode='nearest') for c in range(3)],1)
sw=fin['shirt_w'][:,None]; a=(1-fin['shirt_w']); a[(mat==3)|(mat==4)]=0
P=P*(1-sw)+np.array([0.016,0.018,0.026])*sw; P[mat==3]=[0.06,0.015,0.012]; P[mat==4]=[0.05,0.03,0.025]
CL=np.array([(0.50,0.37,0.29),(0.20,0.155,0.13),(0.07,0.075,0.09),(0.16,0.05,0.05),(0.22,0.15,0.12)])[mat]
hw=fin['hair_w'][:,None]; CL=np.where((mat<=1)[:,None],np.array([0.50,0.37,0.29])*(1-hw)+np.array([0.20,0.155,0.13])*hw,CL)
def vnorm(V,F):
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);N=np.zeros_like(V)
    for i in range(3): np.add.at(N,F[:,i],n)
    return (N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)).astype(np.float32)
bufs=[];views=[];accs=[];off=0
def add(arr,target=None,typ=None,norm=False,minmax=False):
    global off
    a_=np.ascontiguousarray(arr); b=a_.tobytes(); pad=(-len(b))%4; bufs.append(b+b'\0'*pad); v={'buffer':0,'byteOffset':off,'byteLength':len(b)}
    if target: v['target']=target
    views.append(v); off+=len(b)+pad; acc={'bufferView':len(views)-1,'componentType':{np.float32:5126,np.uint32:5125,np.uint16:5123}[a_.dtype.type],'count':int(a_.shape[0]),'type':typ}
    if norm: acc['normalized']=True
    if minmax: acc['min']=a_.min(0).tolist(); acc['max']=a_.max(0).tolist()
    accs.append(acc); return len(accs)-1
c16=lambda C,A: (np.clip(np.c_[C,A],0,1)*65535+0.5).astype(np.uint16)
def prim(V,F,m,C=None,A=None,Cc=None,targets=None):
    V=V.astype(np.float32); N=vnorm(V,F); at={'POSITION':add(V*CM,34962,'VEC3',minmax=True),'NORMAL':add(N,34962,'VEC3')}
    if C is not None: at['COLOR_0']=add(c16(C,A),34962,'VEC4',norm=True); at['_CLAY']=add(c16(Cc,np.zeros(len(V))),34962,'VEC4',norm=True)
    p={'attributes':at,'indices':add(F.astype(np.uint32).ravel(),34963,'SCALAR'),'material':m}
    if targets: p['targets']=[{'POSITION':add((dV*CM).astype(np.float32),34962,'VEC3',minmax=True),'NORMAL':add((vnorm(V+dV,F)-N).astype(np.float32),34962,'VEC3')} for dV in targets]
    return p
tg=[Wv@(an['comps'][k]*an['sd'][k]) for k in range(K)]
meshes=[{'name':'head','primitives':[prim(V,F,0,P,a,CL,tg)],'extras':{'targetNames':['motion %d'%(k+1) for k in range(K)]},'weights':[0]*K}]
for nm in ('eyeR','eyeL'):
    Ve=fin[nm+'_v']; meshes.append({'name':nm,'primitives':[prim(Ve,fin[nm+'_f'],1,fin[nm+'_col'],np.zeros(len(Ve)),np.tile([0.62,0.55,0.48],(len(Ve),1)))]})
for nm,k in (('teethU','tu'),('teethL','tl')): meshes.append({'name':nm,'primitives':[prim(fin[k+'_v'],fin[k+'_f'],2)]})
mats=[{'name':'Skin','pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'metallicFactor':0,'roughnessFactor':0.6}},{'name':'Eye','pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'metallicFactor':0,'roughnessFactor':0.25}},
      {'name':'Teeth','pbrMetallicRoughness':{'baseColorFactor':[0.6,0.55,0.47,1],'metallicFactor':0,'roughnessFactor':0.35}}]
gl={'asset':{'version':'2.0','generator':'scultera take4 (numpy)'},'scene':0,'scenes':[{'nodes':list(range(len(meshes)))}],'nodes':[{'name':m['name'],'mesh':i} for i,m in enumerate(meshes)],'meshes':meshes,'materials':mats,
    'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':off}],'extras':{'eyeR_c':(fin['eyeR_c']*CM).tolist(),'eyeL_c':(fin['eyeL_c']*CM).tolist(),'centre':(sdf4.C*CM).tolist()}}
js=json.dumps(gl,separators=(',',':')).encode(); js+=b' '*((-len(js))%4); bin_=b''.join(bufs)
with open('sam4_raw.glb','wb') as f:
    f.write(struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(bin_))); f.write(struct.pack('<II',len(js),0x4E4F534A)); f.write(js); f.write(struct.pack('<II',len(bin_),0x004E4942)); f.write(bin_)
np.save('work/paint_v.npy',np.c_[P,a].astype(np.float32)); np.save('work/clay_v.npy',CL.astype(np.float32)); print('raw glb %.0f MB'%(len(bin_)/1e6))
