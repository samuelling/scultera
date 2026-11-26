"""Assemble the take-3 time-lapse and its synced log."""
import json, subprocess, glob, os, numpy as np
el=np.load('skull_ell3.npy')
def E(phase,tool,target,note,**p): return dict(phase=phase,tool=tool,target=target,note=note,params=p)
HOLD=1.5
seg=[
 ('s01_points',[E('measure','Track','1,421 frames','47 seconds turning in a chair: the face is found in 871 frames; the rest are the back of the head',faces=871),
                E('measure','Solve pose','745 frames','head pose per frame from its tracked points; frames that don\'t fit are dropped',mean_error_px=2.9,turn='-72° to +66°',nod='-45° to +37°'),
                E('measure','Test','depth','with ±70° of turn the depth is measured, not guessed: the turns agree best at ×1.10',best_scale=1.10)]),
 ('s02_hull_1',[E('carve','Silhouette','1 frame','one frame\'s outline, swept straight back from the camera: a cookie-cutter block',frames=1)]),
 ('s03_hull_6',[E('carve','Intersect','6 frames','six outlines from different angles already make a head',frames=6)]),
 ('s04_hull_30',[E('carve','Intersect','30 frames','profiles cut the nose, lips and chin; the nodding frames cut the crown and under the jaw',frames=30)]),
 ('s05_hull_120',[E('carve','Intersect','120 frames','the ears and jaw corners appear; the back stays a wedge, since the camera never saw it straight on',frames=120)]),
 ('s06_hull_745',[E('carve','Intersect','745 frames','every posed frame: a voxel survives only if at most 3 frames saw background through it',frames=745,voxel_mm=2)]),
 ('s07_back_rounded',[E('blockout','Fit','skull','an ellipsoid fitted to the carved sides and crown (4.5 mm rms)',width_cm=round(2*el[3],1),length_cm=round(2*el[5],1)),
                      E('blockout','Round','back of head','never seen straight on: the fitted skull takes over behind the ears, the carve is smoothed elsewhere',sector='±72°')]),
 ('s08_face_hollows',[E('blockout','Press','face','the tracked face surface pressed into the carve: the carve bounds the outside, the tracked points cut the hollows (eyes, lips, nose wings)',verts=29073)]),
 ('s09_neck_bust',[E('blockout','Add','neck + shoulders','neck, bust and trapezius blended in below the carve',neck_cm=13)]),
 ('s10_ears_nostrils',[E('detail','Clay','ears','ears built out where the carve put them',flare_deg=24),E('detail','Carve','nostrils','drilled at the tracked nostril points',depth_mm=7)]),
 ('s11_lip_border',[E('detail','Pinch','lip border','vermilion edge along the 20-point lip line',radius_mm=0.6)]),
 ('s12_eyes_teeth',[E('finish','Insert','eyes + teeth','eyeballs behind the tracked lids; two tooth arches; mouth slit opened',teeth=24)]),
 ('s13_hair',[E('finish','Mask','hairline','hair vs skin votes from 187 frames',agree='92%'),E('finish','Stipple','buzz cut','raised 2.2 mm with a fine stipple',raise_mm=2.2)]),
 ('s14_shirt',[E('finish','Draw','T-shirt','crew-neck collar band and fabric layer',raise_mm=2.8)]),
 ('s15_paint_1',[E('paint','Project','video colour','249 frames from all around: each posed and expression-matched, hidden points skipped',frames=249,seen='67%')]),
 ('s16_paint_2',[E('paint','Balance','colour','tungsten room light neutralised; broad lighting flattened',passes=2)]),
 ('s17_paint_3',[E('paint','Fill','unseen','the rest filled from what was seen',fill='33%')]),
]
frames=[]; T=0.0; entries=[]
for still,es in seg:
    dt=HOLD/len(es)
    for i,e in enumerate(es):
        e['t']=round(T+i*dt,2)
        if not e['note']: e['note']='a hundred and twenty outlines'
        entries.append(e)
    frames.append(('still',f'tl/{still}.png',HOLD)); T+=HOLD
entries.append(dict(E('finish','Turntable','finished bust','a full turn of the painted bust'),t=round(T,2)))
nt=len(glob.glob('tl/turn_*.png')); frames.append(('seq','tl/turn_%03d.png',nt,12)); T+=nt/12
npf=len(glob.glob('tl/perf_*.png'))
pe=[E('motion','Transfer','expressions','the same face map ties every vertex to the tracked points (35,392 move)',moving_verts=35392),
    E('motion','Replay','the talking clip','take 2\'s 12-second clip, performed by the take-3 head',shapes=12)]
for i,e in enumerate(pe): e['t']=round(T+i*(npf/10)/len(pe),2); entries.append(e)
frames.append(('seq','tl/perf_%03d.png',npf,10)); T+=npf/10
os.makedirs('vid',exist_ok=True); parts=[]
for k,fr in enumerate(frames):
    out=f'vid/p{k:02d}.mp4'
    if fr[0]=='still': cmd=['ffmpeg','-y','-v','error','-loop','1','-t',str(fr[2]),'-i',fr[1],'-r','30','-vf','scale=560:560,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','22',out]
    else: cmd=['ffmpeg','-y','-v','error','-framerate',str(fr[3]),'-i',fr[1],'-r','30','-vf','scale=560:560,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','22',out]
    subprocess.run(cmd,check=True); parts.append(out)
open('vid/list.txt','w').write(''.join(f"file '{os.path.basename(p)}'\n" for p in parts))
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i','vid/list.txt','-c','copy','-movflags','+faststart','sculpt3.mp4'],check=True)
subprocess.run(['ffmpeg','-y','-v','error','-i','tl/s17_paint_3.png','-vf','scale=560:560','-q:v','4','poster3.jpg'],check=True)
json.dump({'entries':entries,'duration':round(T,2)},open('timeline3.json','w'),indent=1)
print('video %.1fs, %d log entries'%(T,len(entries)))
