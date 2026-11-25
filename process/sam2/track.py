import mediapipe as mp, numpy as np, glob
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
fs=sorted(glob.glob('frames/f*.jpg'))
opt=vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path='models/face_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,output_face_blendshapes=True,output_facial_transformation_matrixes=True,num_faces=1)
det=vision.FaceLandmarker.create_from_options(opt)
sopt=vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path='models/selfie_multiclass.tflite'),output_category_mask=True)
seg=vision.ImageSegmenter.create_from_options(sopt)
LM=[];M=[];BS=[];ok=[];masks=[];names=None
for i,f in enumerate(fs):
    img=mp.Image.create_from_file(f)
    r=det.detect_for_video(img,int(i*1000/30))
    if not r.face_landmarks: ok.append(False);LM.append(np.zeros((478,3)));M.append(np.eye(4));BS.append(np.zeros(52));masks.append(np.zeros((img.height,img.width),np.uint8));continue
    ok.append(True)
    LM.append(np.array([[p.x*img.width,p.y*img.height,p.z*img.width] for p in r.face_landmarks[0]]))
    M.append(np.array(r.facial_transformation_matrixes[0]))
    BS.append(np.array([b.score for b in r.face_blendshapes[0]]));names=[b.category_name for b in r.face_blendshapes[0]]
    masks.append(seg.segment(img).category_mask.numpy_view().copy())
np.savez_compressed('track.npz',lm=np.array(LM),M=np.array(M),bs=np.array(BS),ok=np.array(ok),bsnames=np.array(names),masks=np.array(masks),W=img.width,H=img.height)
print(sum(ok),'/',len(fs),'tracked')
