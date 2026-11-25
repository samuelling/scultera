import sys; sys.path.insert(0,'proc')
import bpy, numpy as np
from bl_common import *
out=sys.argv[1]; views=sys.argv[2].split(',') if len(sys.argv)>2 else ['front','q','side']
sc=reset(); (load_painted() if 'paint' in out else load_finished()); studio(sc); cycles(sc,420,520,int(sys.argv[3]) if len(sys.argv)>3 else 12)
V={'front':((0,-105,-2),(0,6,-3)),'q':((-58,-86,3),(0,6,-3)),'side':((-105,6,-2),(0,6,-3)),'low':((10,-92,-24),(0,0,-1))}
import subprocess
files=[]
for v in views:
    for o in list(bpy.data.objects):
        if o.type=='CAMERA': bpy.data.objects.remove(o)
    camera(sc,*V[v],lens=85); sc.render.filepath=f'/tmp/pv_{v}.png'; bpy.ops.render.render(write_still=True); files.append(f'/tmp/pv_{v}.png')
from PIL import Image
ims=[Image.open(f) for f in files]; c=Image.new('RGB',(sum(i.width for i in ims),ims[0].height))
x=0
for i in ims: c.paste(i,(x,0)); x+=i.width
c.save(out)
