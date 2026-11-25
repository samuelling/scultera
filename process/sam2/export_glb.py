"""Write the take-2 bust as glTF directly (Y-up, metres): head with paint + clay colours and 13 morph targets, eyes, teeth."""
import numpy as np, json, struct
from scipy import sparse as sp
fin=np.load('finished.npz'); an=np.load('anim.npz'); Wv=sp.load_npz('Wv.npz')
P=np.load('paint.npy'); K=int(an['K'])
CM=0.01
def vnorm(V,F):
    n=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);N=np.zeros_like(V)
    for i in range(3): np.add.at(N,F[:,i],n)
    return (N/np.maximum(np.linalg.norm(N,axis=1,keepdims=True),1e-12)).astype(np.float32)
from bl_colors import CLAYC
bufs=[]; views=[]; accs=[]; off=0
def add(arr,target=None,typ=None,comp=None,norm=False,minmax=False):
    global off
    a=np.ascontiguousarray(arr); b=a.tobytes(); pad=(-len(b))%4
    bufs.append(b+b'\0'*pad); v={'buffer':0,'byteOffset':off,'byteLength':len(b)}
    if target: v['target']=target
    views.append(v); off+=len(b)+pad
    ct={np.float32:5126,np.uint32:5125,np.uint16:5123,np.uint8:5121}[a.dtype.type]
    acc={'bufferView':len(views)-1,'componentType':ct,'count':int(a.shape[0]),'type':typ}
    if norm: acc['normalized']=True
    if minmax: acc['min']=a.min(0).tolist(); acc['max']=a.max(0).tolist()
    accs.append(acc); return len(accs)-1
def col16(C): return (np.clip(np.c_[C,np.ones(len(C))],0,1)*65535+0.5).astype(np.uint16)
meshes=[]; nodes=[]
mats=[{'name':'Skin','pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'metallicFactor':0,'roughnessFactor':0.58}},
      {'name':'Eye','pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'metallicFactor':0,'roughnessFactor':0.28}},
      {'name':'Teeth','pbrMetallicRoughness':{'baseColorFactor':[0.74,0.69,0.6,1],'metallicFactor':0,'roughnessFactor':0.35}}]
def prim(V,F,mat,C=None,Cc=None,targets=None):
    V=V.astype(np.float32); N=vnorm(V,F)
    at={'POSITION':add(V*CM,34962,'VEC3',minmax=True),'NORMAL':add(N,34962,'VEC3')}
    if C is not None: at['COLOR_0']=add(col16(C),34962,'VEC4',norm=True)
    if Cc is not None: at['_CLAY']=add(col16(Cc),34962,'VEC4',norm=True)
    p={'attributes':at,'indices':add(F.astype(np.uint32).ravel(),34963,'SCALAR'),'material':mat}
    if targets:
        p['targets']=[]
        for dV in targets:
            dN=vnorm(V+dV,F)-N
            p['targets'].append({'POSITION':add((dV*CM).astype(np.float32),34962,'VEC3',minmax=True),'NORMAL':add(dN.astype(np.float32),34962,'VEC3')})
    return p
V=fin['v'].astype(np.float64); F=fin['f']; mat=fin['mat']
comps=an['comps']; coef=an['coef']; sd=coef.std(0)
tg=[Wv@comps[0]]+[Wv@(comps[k+1]*sd[k]) for k in range(K)]
names=['clip offset']+['motion %d'%(k+1) for k in range(K)]
Cc=CLAYC[mat]
meshes.append({'name':'head','primitives':[prim(V,F,0,P,Cc,tg)],'extras':{'targetNames':names},'weights':[0]*len(tg)})
def eye_cols(Vv,c):
    d=Vv-c; d/=np.linalg.norm(d,axis=1,keepdims=True); ang=np.degrees(np.arccos(np.clip(d[:,2],-1,1)))
    sclera=np.array([0.36,0.31,0.27]); iris=np.array([0.035,0.022,0.016]); pupil=np.array([0.008,0.006,0.005])
    t=np.clip((ang-24)/3,0,1)[:,None]; col=iris*(1-t)+sclera*t; return np.where((ang<9)[:,None],pupil,col)
for nm in ('eyeR','eyeL'):
    Vv=fin[nm+'_v']; meshes.append({'name':nm,'primitives':[prim(Vv,fin[nm+'_f'],1,eye_cols(Vv,fin[nm+'_c']),np.tile([0.80,0.74,0.68],(len(Vv),1)))]})
for nm,k in (('teethU','tu'),('teethL','tl')):
    meshes.append({'name':nm,'primitives':[prim(fin[k+'_v'],fin[k+'_f'],2)]})
nodes=[{'name':m['name'],'mesh':i} for i,m in enumerate(meshes)]
gl={'asset':{'version':'2.0','generator':'scultera take2 (numpy)'},'scene':0,'scenes':[{'nodes':list(range(len(nodes)))}],'nodes':nodes,'meshes':meshes,
    'materials':mats,'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':off}],
    'extras':{'eyeR_c':(fin['eyeR_c']*CM).tolist(),'eyeL_c':(fin['eyeL_c']*CM).tolist()}}
js=json.dumps(gl,separators=(',',':')).encode(); js+=b' '*((-len(js))%4); bin_=b''.join(bufs)
with open('sam2_raw.glb','wb') as f:
    f.write(struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(bin_))); f.write(struct.pack('<II',len(js),0x4E4F534A)); f.write(js)
    f.write(struct.pack('<II',len(bin_),0x004E4942)); f.write(bin_)
# per-frame curves for the browser: 30 fps, coefficient for each target, head quaternion, gaze, jaw
curves={'fps':30,'frames':int(len(coef)),'coef':np.round(coef/sd,3).tolist(),'quat':np.round(an['quat'],4).tolist(),
        'gaze':np.round(an['gaze'],1).tolist(),'jaw':np.round(an['jaw']*CM,5).tolist(),'K':K}
json.dump(curves,open('motion.json','w'),separators=(',',':'))
print('raw glb %.1f MB, motion.json %.0f kB'%(len(bin_)/1e6,len(json.dumps(curves))/1e3))
