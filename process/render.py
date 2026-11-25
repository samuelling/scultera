import bpy,sys
a=sys.argv[sys.argv.index('--')+1:];cam,out,res,spp=a[0],a[1],float(a[2]),int(a[3])
bpy.ops.wm.open_mainfile(filepath='/home/claude/gecko/sad_gecko.blend')
sc=bpy.context.scene;sc.camera=bpy.data.objects['Cam_'+cam];sc.render.resolution_percentage=int(res*100);sc.cycles.samples=spp
sc.render.filepath=out;bpy.ops.render.render(write_still=True)
