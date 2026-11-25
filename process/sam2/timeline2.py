"""Assemble the take-2 time-lapse (ffmpeg) and its synced log."""
import json, subprocess, glob, os, numpy as np
B=json.load(open('log_build.json')); Fn=json.load(open('log_finish.json'))
sk=np.load('skull_fit.npy'); an=np.load('anim.npz')
from scipy.spatial.transform import Rotation as Rot
eu=Rot.from_quat(an['quat']).as_euler('xyz',degrees=True)
def E(phase,tool,target,note,**p): return dict(phase=phase,tool=tool,target=target,note=note,params=p)
HOLD=1.6
seg=[  # (still, entries)
 ('s01_points_frame1',[E('measure','Track','358 video frames','MediaPipe finds 478 points and 52 expression scores on every frame of the 12-second clip',tracked='358/358',fps=30)]),
 ('s02_points_30',[E('measure','Solve pose','head per frame','each frame\'s head rotation and position solved from its 2D points (perspective, 63° lens)',mean_error_px=4.7,turn_deg='%d to %d'%(eu[:,1].min(),eu[:,1].max()))]),
 ('s03_points_all',[E('measure','Average','resting face','frames are averaged point by point, each region weighted toward the frames where it is calm',frames=358),
                    E('measure','Test','depth','which depth scale do the head turns agree with? x1.03 fits best, so the averaged depth stands',best_scale=1.03),
                    E('measure','Symmetrize','midline','symmetry plane found (3.4° off); 25% of his own asymmetry kept',kept='25%'),
                    E('measure','Set','rest pose','eyes at his median opening (6.4 mm), lips closed',eye_mm=6.4)]),
 ('s04_face_surface',[E('measure','Subdivide','face surface','the 898-triangle landmark mesh, Loop-subdivided three times; eyes and mouth left open',verts=29073)]),
 ('s05_skull',[E('blockout','Fit','skull','an ellipsoid fitted to his outline in 30 frames (sides, crown); 31 px error before, 12 after',width_cm=round(2*sk[0],1),length_cm=round(2*sk[2],1))]),
 ('s06_face_on_skull',[E('blockout','Project','face','the measured face laid onto the skull along rays from the head centre',rays='0.2° grid'),
                       E('blockout','Conform','face edge','the skull is nudged to meet the face edge, so there is no seam or step',blend_deg=16)]),
 ('s07_neck_bust',[E('blockout','Add','jaw + neck','under-jaw mass and a thick neck, smooth-unioned',neck_cm=14),E('blockout','Add','shoulders','bust block with trapezius slopes, cut flat',width_cm=35)]),
 ('s08_ears',[E('blockout','Add','ears','helix rim and concha bowl, flared 24°, checked against his outline in the frames',flare_deg=24)]),
 ('s09_nostrils',[B[-4] if False else E('detail','Carve','nostrils','drilled where the tracked nostril points sit, angled down and forward',depth_mm=7)]),
 ('s10_lids',[E('detail','Clay','upper lids','lid margins thickened along the tracked eyelid rims',radius_mm=0.9)]),
 ('s11_brows',[E('detail','Clay','brows','a low brow ridge under the tracked brow points',radius_mm=2)]),
 ('s12_lip_border',[E('detail','Pinch','lip border','crisp vermilion edge traced along the 20-point outer lip line',radius_mm=0.6)]),
 ('s13_eyes_teeth',[E('finish','Carve','mouth','slit along the inner-lip line, opened 13 mm deep so the jaw can drop',depth_mm=13),
                    E('finish','Insert','eyeballs','2.4 cm spheres set behind each 16-point eyelid rim, irises incised',radius_mm=12),
                    E('finish','Insert','teeth','two arches; the lower one rides the chin',teeth=24)]),
 ('s14_hair',[E('finish','Mask','hairline','hairline curve checked against hair/skin votes from 90 frames',agree='91%'),E('finish','Stipple','buzz cut','hair cap raised 2.2 mm with a fine stipple',raise_mm=2.2)]),
 ('s15_shirt',[E('finish','Draw','T-shirt','crew-neck collar band and a 3 mm fabric layer',raise_mm=2.8)]),
 ('s16_paint_1',[E('paint','Project','video colour','179 frames projected onto the sculpt, each posed and expression-matched first, hidden points skipped',frames=179)]),
 ('s17_paint_2',[E('paint','Even out','lighting','the window lit one cheek and the lamp the other; broad light flattened, detail kept',passes=2)]),
 ('s18_paint_3',[E('paint','Fill','unseen skin','sides and back the camera never saw, filled outward from what it did see',seen='31%')]),
]
frames=[]; T=0.0; entries=[]
for still,es in seg:
    dt=HOLD/len(es)
    for i,e in enumerate(es): e['t']=round(T+i*dt,2); entries.append(e)
    frames.append(('still',f'tl/{still}.png',HOLD)); T+=HOLD
entries.append(dict(E('finish','Turntable','finished bust','a full turn of the painted bust at rest'),t=round(T,2)))
nt=len(glob.glob('tl/turn_*.png')); frames.append(('seq','tl/turn_%03d.png',nt,12)); T+=nt/12
pe=[E('motion','Transfer','expressions','every sculpt vertex follows the tracked points through the face map (35,722 vertices move)',moving_verts=35722),
    E('motion','Compress','12 motion shapes','PCA of the clip\'s 358 frames of point motion: 12 shapes cover 98%',shapes=12),
    E('motion','Replay','head + gaze','head rotation and eye direction from every frame',turn_deg='%d to %d'%(eu[:,1].min(),eu[:,1].max())),
    E('motion','Replay','the clip','the 12 seconds performed by the sculpt, in real time')]
npf=len(glob.glob('tl/perf_*.png'))
for i,e in enumerate(pe): e['t']=round(T+i*(npf/10)/len(pe),2); entries.append(e)
frames.append(('seq','tl/perf_%03d.png',npf,10)); T+=npf/10
# ffmpeg: build each piece as a clip, then concat
os.makedirs('vid',exist_ok=True); parts=[]
for k,fr in enumerate(frames):
    out=f'vid/p{k:02d}.mp4'
    if fr[0]=='still':
        cmd=['ffmpeg','-y','-v','error','-loop','1','-t',str(fr[2]),'-i',fr[1],'-r','30','-vf','scale=560:560,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','22',out]
    else:
        cmd=['ffmpeg','-y','-v','error','-framerate',str(fr[3]),'-i',fr[1],'-r','30','-vf','scale=560:560,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','22',out]
    subprocess.run(cmd,check=True); parts.append(out)
open('vid/list.txt','w').write(''.join(f"file '{os.path.basename(p)}'\n" for p in parts))
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i','vid/list.txt','-c','copy','-movflags','+faststart','sculpt2.mp4'],check=True)
subprocess.run(['ffmpeg','-y','-v','error','-i','tl/s18_paint_3.png','-vf','scale=560:560','-q:v','4','poster2.jpg'],check=True)
json.dump({'entries':entries,'duration':round(T,2)},open('timeline2.json','w'),indent=1)
print('video %.1fs, %d log entries'%(T,len(entries)))
