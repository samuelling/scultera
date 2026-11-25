import sys; sys.path.insert(0,'proc')
import bpy, numpy as np
from bl_common import *
which=sys.argv[1]; out=sys.argv[2]
sc=reset(); studio(sc); cycles(sc,560,560,24)
if which=='t1':
    d=np.load('/home/claude/samhead/final.npz'); V1=d['v']; F1=d['f']
    # take-1 data is already blender-style (x, depth, up); align its eye line / face centre to take 2's frame
    V=np.c_[V1[:,0],V1[:,2],-V1[:,1]]
    S1=np.load('/home/claude/samhead/aligned.npz')['ident']; e=(S1[468]+S1[473])/2; V=V-np.array([e[0],e[2],-e[1]])
    add_mesh('t1',V,F1,[mk_mat('clay1',(0.5,0.47,0.44),0.6)])
elif which=='clay': load_finished()
else: load_painted()
camera(sc,(-14,-70,1.5),(0,0,-1.6),lens=70)
sc.render.filepath=out; bpy.ops.render.render(write_still=True)
