from pathlib import Path
import hashlib, json, math, struct
import cv2,numpy as np
root=Path.cwd(); out=root/'native/src/color_lab_tables.h'
assert cv2.__version__=='4.8.0'
r,g,b=np.meshgrid(np.arange(33,dtype='f')/32,np.arange(33,dtype='f')/32,np.arange(33,dtype='f')/32,indexing='ij')
grid=np.stack([b,g,r],axis=-1).reshape(33*33,33,3)
lab=cv2.cvtColor(grid,cv2.COLOR_BGR2LAB)
lut=np.rint(np.stack([lab[:,:,0]*np.float32(16384/100),(lab[:,:,1]+128)*64,(lab[:,:,2]+128)*64],axis=-1)).astype(np.int16)
reconstruct=np.stack([lut[:,:,0].astype('f')*np.float32(100/16384),lut[:,:,1].astype('f')*np.float32(1/64)-128,lut[:,:,2].astype('f')*np.float32(1/64)-128],axis=-1)
assert np.array_equal(lab.view('u4'),reconstruct.view('u4'))
f=np.float32
samples=np.array([x* (323/25) if x<=7827/2500000 else math.pow(x,1/(12/5))*(1+11/200)-11/200 for x in np.arange(1025,dtype='d')/1024],dtype='f')
table=np.zeros((1024,4),dtype='f');cn=f(0)
for i in range(1,1024):
 t=f(f(f(samples[i+1]-f(samples[i]*f(2)))+samples[i-1])*f(3))
 l=f(f(1)/f(f(4)-table[i-1,0]));table[i,0]=l;table[i,1]=f(f(t-table[i-1,1])*l)
for i in range(1023,-1,-1):
 c=f(table[i,1]-f(table[i,0]*cn));bb=f(f(samples[i+1]-samples[i])-f(f(cn+f(c*f(2)))/f(3)));d=f(f(cn-c)/f(3))
 table[i]=[samples[i],bb,c,d];cn=c
coefs=np.array([[3.240479,-1.53715,-.498535],[-.969256,1.875991,.041556],[.055648,-.204043,1.057311]],'d')*np.array([.950456,1,1.088754],'d')
lines=['/* Immutable constants of OpenCV 4.8.0 color_lab.cpp (Apache-2.0).', ' * RGB lattice: p=B/32, q=G/32, r=R/32; flattened p fastest.', ' * Grid fractions are exactly zero, so the original trilinear interpolator', ' * returns its own int16 lattice entries without fitting or approximation.', ' * Generation input/source hashes: native/reports/color-table-generation.json.', ' * Inverse gamma samples and natural cubic spline follow applyInvGamma and', ' * splineBuild, rounding every softfloat operation to IEEE float32. */', '#ifndef CHAINNER_COLOR_LAB_TABLES_H','#define CHAINNER_COLOR_LAB_TABLES_H','static const int16_t lab_lattice[33 * 33 * 33 * 3] = {']
a=lut.ravel()
lines += ['    '+', '.join(str(v) for v in a[i:i+18])+',' for i in range(0,len(a),18)]
lines += ['};','static const float lab_inverse_gamma[1024 * 4] = {']
a=table.ravel();lines += ['    '+', '.join(float(v).hex()+'f' for v in a[i:i+4])+',' for i in range(0,len(a),4)]
lines += ['};','static const float lab_xyz_to_rgb[9] = {', '    '+', '.join(float(f(v)).hex()+'f' for v in coefs.ravel()),'};','#endif','']
out.write_text('\n'.join(lines),encoding='utf-8')
manifest={'opencv':cv2.__version__,'source':'https://github.com/opencv/opencv/blob/4.8.0/modules/imgproc/src/color_lab.cpp','source_sha256':hashlib.sha256((root/'native/reports/reference-sources-color/color_lab.cpp').read_bytes()).hexdigest(),'grid_shape':list(grid.shape),'grid_sha256':hashlib.sha256(grid.tobytes()).hexdigest(),'lattice_sha256':hashlib.sha256(lut.tobytes()).hexdigest(),'grid_reconstruction_bit_exact':True,'inverse_gamma_samples_sha256':hashlib.sha256(samples.tobytes()).hexdigest(),'inverse_gamma_spline_sha256':hashlib.sha256(table.tobytes()).hexdigest(),'header_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(root/'native/reports/color-table-generation.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
