"""Assemble the take-4 time-lapse and its synced log."""
import json, subprocess, glob, os
def E(phase,tool,target,note,**p): return dict(phase=phase,tool=tool,target=target,note=note,params=p)
seg=[
 ('s01_carve',4.2,[E('pose','Reuse','745 tracked frames','take 3\'s face tracking and per-frame head poses are kept as they were',frames=745),
   E('pose','Measure','chair axis','the first tracked turn (62°) gives the axis the chair spins about',tilt_deg=5),
   E('pose','Flow','back-of-head turns','no face to track from behind, so the turn is read from how the hair moves frame to frame, separating turn from slide',frames=676),
   E('pose','Anchor','exact back views','when both ears show equal skin either side of the hair, he is seen from exactly behind: six such instants pin the turn',anchors=6),
   E('pose','Check','silhouettes','outline agreement is flat across flow scales (0.763 to 0.776), so the ears, not the outline, set the scale',scale=1.15),
   E('carve','Sharpen','masks','each frame\'s person mask snapped to image edges with a guided filter',radius_px=7),
   E('carve','Intersect','745 outlines','the head carved on a 2 mm grid; back views would trim only 1% more, so they are left out',voxel_mm=2)]),
 ('s02_smoothed',2.0,[E('surface','Smooth','scalp','the carve is streaky on the scalp from pose noise: smoothed harder there than on the ears and nose',sigma_mm=7.5),
   E('surface','Round','back ridge','±70° of tracked turn leaves a ridge of at most 6 mm down the back: rounded off',max_mm=6),
   E('surface','Trim','shoulders','shoulders are not rigid with the head, so only the neck column is kept below the jaw',cut_cm=-17)]),
 ('s03_face',2.4,[E('surface','Conform','face rim','tracked face and carve disagree by -6 to +11 mm at the rim; the face is shifted to meet the carve there',max_mm=11),
   E('surface','Cut','hollows','inside the face the tracked points cut what an outline cannot see: eye sockets, lips, nose wings',verts=29073),
   E('experiment','Plane sweep','stereo relief','15 frame pairs 14° apart, two independent sets: their relief maps agree at r = 0.39. Too noisy: not used',r=0.39,rms_mm=1.3)]),
 ('s04_bust',1.4,[E('surface','Add','neck + bust','the carve\'s own neck runs into a bust block with trapezius slopes',width_cm=35)]),
 ('s05_details',1.4,[E('detail','Carve','nostrils, ears','nostrils at the tracked points; a concha hollow where the carve put each ear',depth_mm=7),E('detail','Pinch','lip border','vermilion edge along the 20-point lip line',radius_mm=0.6)]),
 ('s06_finish',1.8,[E('finish','Raise','buzz cut','hair raised 1.6 mm with stipple where the frames, and the paint\'s own darkness, say hair',raise_mm=1.6),
   E('finish','Add','shirt, teeth, eyes','collar band and fabric layer; two tooth arches; 2.4 cm eyeballs behind the lids',teeth=24)]),
 ('s07_paint_1',1.6,[E('paint','Choose','13 key frames','slow, calm frames only: one frontal, four sides, below, above, six from behind',frames=13),
   E('paint','Project','each key frame','the surface is given that frame\'s expression, then the frame is projected on, hidden texels skipped',texels='2048 x 1024')]),
 ('s08_paint_2',1.6,[E('paint','Register','view to view','each new view is colour-matched to what is painted, then nudged into register by optical flow (6 to 10 texels)',max_texels=14),
   E('paint','Blend','two bands','broad colour blended widely, fine detail taken from the single best view, so it stays sharp',split_texels=8)]),
 ('s09_paint_3',1.8,[E('paint','Close','the loop at the back','going round the head both ways leaves two exposures meeting at the back: cross-faded, hair tone evened',seen='79%'),
   E('paint','Paint','eyeballs','irises and whites sampled from the frontal frame through its own eyelid outline',frame='45.4 s')]),
]
frames=[]; T=0.0; entries=[]
for still,hold,es in seg:
    for i,e in enumerate(es): e['t']=round(T+i*hold/len(es),2); entries.append(e)
    frames.append(('still','tl/%s.png'%still,hold)); T+=hold
entries.append(dict(E('finish','Turntable','finished bust','a full turn of the painted bust'),t=round(T,2)))
nt=len(glob.glob('tl/turn_*.png')); frames.append(('seq','tl/turn_%03d.png',nt,12)); T+=nt/12
npf=len(glob.glob('tl/perf_*.png'))
pe=[E('motion','Transfer','expressions','43,461 vertices follow the tracked points through the face map',moving_verts=43461),
    E('motion','Compress','14 motion shapes','PCA of the tracked face motion keeps 96.8%',shapes=14),
    E('motion','Split','chair and neck','each frame\'s head rotation = a turn about the chair axis (up to ±200°) + what the neck did (up to 48°)',turn_deg=200),
    E('motion','Replay','the first 20 seconds','the sculpt performs the clip, spins included; the page plays all 47 seconds live')]
for i,e in enumerate(pe): e['t']=round(T+i*(npf/10)/len(pe),2); entries.append(e)
frames.append(('seq','tl/perf_%03d.png',npf,10)); T+=npf/10
os.makedirs('vid',exist_ok=True); parts=[]
for k,fr in enumerate(frames):
    out='vid/p%02d.mp4'%k; common=['-r','30','-vf','scale=560:560,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','22',out]
    cmd=['ffmpeg','-y','-v','error','-loop','1','-t',str(fr[2]),'-i',fr[1]]+common if fr[0]=='still' else ['ffmpeg','-y','-v','error','-framerate',str(fr[3]),'-i',fr[1]]+common
    subprocess.run(cmd,check=True); parts.append(out)
open('vid/list.txt','w').write(''.join("file '%s'\n"%os.path.basename(p) for p in parts))
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i','vid/list.txt','-c','copy','-movflags','+faststart','sculpt4.mp4'],check=True)
subprocess.run(['ffmpeg','-y','-v','error','-i','tl/s09_paint_3.png','-vf','scale=560:560','-q:v','4','poster4.jpg'],check=True)
json.dump({'entries':entries,'duration':round(T,2)},open('timeline4.json','w'),indent=1); print('video %.1fs, %d log entries'%(T,len(entries)))
