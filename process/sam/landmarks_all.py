import mediapipe as mp, numpy as np, json, glob, os
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
opt=vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path='models/face_landmarker.task'),output_face_blendshapes=True,output_facial_transformation_matrixes=True,num_faces=3)
det=vision.FaceLandmarker.create_from_options(opt)
out={}
for f in sorted(glob.glob('ref/p0*.jpg')):
    img=mp.Image.create_from_file(f);r=det.detect(img);nm=os.path.basename(f)[:-4]
    if not r.face_landmarks: print(nm,'NO FACE');continue
    # pick largest face
    k=int(np.argmax([np.ptp([p.x for p in fl]) for fl in r.face_landmarks]))
    L=np.array([[p.x*img.width,p.y*img.height,p.z*img.width] for p in r.face_landmarks[k]])
    M=np.array(r.facial_transformation_matrixes[k]);bs={b.category_name:float(b.score) for b in r.face_blendshapes[k]}
    out[nm]=dict(lm=L.tolist(),M=M.tolist(),bs=bs)
    top=sorted(bs.items(),key=lambda t:-t[1])[:5]
    print(nm,len(r.face_landmarks),'faces; px ipd %.0f'%np.linalg.norm(L[468]-L[473]),[(a,round(b,2)) for a,b in top])
json.dump(out,open('lm_all.json','w'))
