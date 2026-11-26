import numpy as np,sys; sys.path.insert(0,'proc'); from prev import sheet
m=np.load(sys.argv[1]); sheet([(m['v'],m['f'],(0.78,0.68,0.6))],sys.argv[2],views=((0,0),(30,-10),(90,0),(160,0)),size=420,span=float(sys.argv[3]) if len(sys.argv)>3 else 36,center=(0,-7,-7))
