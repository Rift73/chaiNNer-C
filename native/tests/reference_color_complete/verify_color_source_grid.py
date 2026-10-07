"""Independent OpenCV Lab lattice initializer cross-check.

Altered Python translation of the positive finite f32_cbrt routine.
Original source notices follow:
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on files from packages softfloat and fdlibm
// issued with the following licenses:

/*============================================================================

This C source file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3c, by John R. Hauser.

Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
University of California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

// FDLIBM licenses:

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


"""
from pathlib import Path
import ctypes as ct,struct,numpy as np,cv2
f=np.float32;fm=ct.CDLL('ucrtbase.dll').fmaf;fm.argtypes=[ct.c_float]*3;fm.restype=ct.c_float
A=[struct.unpack('<d',struct.pack('<Q',v))[0] for v in [0x4046a09e6653ba70,0x406808f46c6116e0,0x405dca97439cae14,0x402add70d2827500,0x3fc4f15f83f55d2d,0x402d9e20660edb21,0x4062ff15c0285815,0x406510d06a8112ce,0x4040fecbc9e2c375,0x3ff0000000000000]]
def cubert(x):
 bits=int(np.float32(x).view('u4'));ex=((bits>>23)&255)-127;shx=ex%3 if ex>=0 else -((-ex)%3);shx-=3 if shx>=0 else 0;ex=(ex-shx)//3-1
 fr=struct.unpack('<d',struct.pack('<Q',((shx+1023)<<52)|((bits&0x7fffff)<<29)))[0]
 fr=((((A[0]*fr+A[1])*fr+A[2])*fr+A[3])*fr+A[4])/((((A[5]*fr+A[6])*fr+A[7])*fr+A[8])*fr+A[9])
 fb=struct.unpack('<Q',struct.pack('<d',fr))[0];out=((ex+127)<<23)|((fb&0xfffffffffffff)>>29)
 return np.uint32(out if bits&0x7fffffff else 0).view('f')
r,g,b=np.meshgrid(np.arange(33,dtype='f')/32,np.arange(33,dtype='f')/32,np.arange(33,dtype='f')/32,indexing='ij');grid=np.stack([b,g,r],-1).reshape(1089,33,3);reference=cv2.cvtColor(grid,cv2.COLOR_BGR2LAB)
linear=np.array([float(x)/12.92 if x<=809/20000 else ((float(x)+11/200)/(1+11/200))**(12/5) for x in np.arange(33,dtype='f')/32],dtype='f')
C=np.array([[.412453,.357580,.180423],[.212671,.715160,.072169],[.019334,.119193,.950227]],'d')/np.array([.950456,1,1.088754],'d')[:,None];C=C.astype('f')[:,::-1]
lscale=f(f(841)/f(108));lbias=f(f(16)/f(116));threshold=f(f(216)/f(24389));out=np.empty_like(reference)
for index,(bb,gg,rr) in enumerate((grid.reshape(-1,3)*32).astype(int)):
 v=linear[[bb,gg,rr]];xyz=[f(f(v[0]*c[0]+v[1]*c[1])+v[2]*c[2]) for c in C];fx,fy,fz=[cubert(x) if x>threshold else f(fm(float(x),float(lscale),float(lbias))) for x in xyz]
 L=f(f(116)*fy-f(16)) if xyz[1]>threshold else f(f(f(29*29*29)/f(27))*xyz[1]);a=f(f(500)*f(fx-fy));bv=f(f(200)*f(fy-fz))
 il=round(float(f(f(f(16384)*L)/f(100))));ia=round(float(f(f(f(16384)*f(a+f(128)))/f(256))));ib=round(float(f(f(f(16384)*f(bv+f(128)))/f(256))))
 out.reshape(-1,3)[index]=[f(il)*f(100/16384),f(ia)*f(1/64)-f(128),f(ib)*f(1/64)-f(128)]
print('Grid elements',out.size,'mismatch',np.count_nonzero(out.view('u4')!=reference.view('u4')))
np.save('native/reports/color_source_initialized_grid.npy',out)
