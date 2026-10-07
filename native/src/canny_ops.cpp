/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
/* Canny's integer suppression rules adapted from OpenCV 5.0.0 canny.cpp.
 * The complete original Intel/Itseez notice is retained below by the source
 * import step. No OpenCV runtime code is called. The admitted node contract is
 * the default 3x3 Sobel aperture and L1 magnitude, with multi-channel maximum. */
#include "repair_buffers.hpp"
#include "parallel.h"
#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
struct Derivatives {
    const uint8_t *source;
    size_t height,width,channels;
    int *dx,*dy,*magnitude;
};
void derivatives(void *context,size_t begin,size_t end) noexcept {
    auto &j=*static_cast<Derivatives *>(context);
    size_t pitch=j.width+2;
    for(size_t y=begin;y<end;++y) {
        size_t above=y?y-1:y,below=y+1<j.height?y+1:y;
        for(size_t x=0;x<j.width;++x) {
            size_t left=x?x-1:x,right=x+1<j.width?x+1:x;
            int magnitude=-1,dx=0,dy=0;
            for(size_t c=0;c<j.channels;++c) {
                auto pixel=[&](size_t r,size_t s) { return static_cast<int>(j.source[(r*j.width+s)*j.channels+c]); };
                int gx=(pixel(above,right)-pixel(above,left))+2*(pixel(y,right)-pixel(y,left))+(pixel(below,right)-pixel(below,left));
                int gy=(pixel(below,left)-pixel(above,left))+2*(pixel(below,x)-pixel(above,x))+(pixel(below,right)-pixel(above,right));
                int m=std::abs(gx)+std::abs(gy);
                if(m>magnitude) { magnitude=m;dx=gx;dy=gy; }
            }
            size_t index=y*j.width+x;
            j.dx[index]=dx;j.dy[index]=dy;
            j.magnitude[(y+1)*pitch+x+1]=magnitude;
        }
    }
}
}

extern "C" CN_EXPORT cn_status cn_canny_u8_policy(const uint8_t *source,uint8_t *output,
        size_t height,size_t width,size_t channels,double lower,double upper,int ipp_thresholds) noexcept {
    size_t pixels;
    cn_status status=cn_repair::dimensions(height,width,channels,pixels);
    if(status!=CN_OK) return status;
    // OpenCV 5.0.0's CV_CN_MAX (core/hal/interface.h); cv2.Canny rejects more.
    if(channels>128 || ipp_thresholds<0 || ipp_thresholds>1) return CN_INVALID_ARGUMENT;
    cn_repair::Region input{},out{};
    status=cn_repair::region(source,pixels*channels,input);if(status!=CN_OK)return status;
    status=cn_repair::region(output,pixels,out);if(status!=CN_OK)return status;
    if(cn_repair::overlap(input,out))return CN_INVALID_ARGUMENT;
    try {
        if(lower>upper)std::swap(lower,upper);
        int low=cn_repair::floor_int(lower),high=cn_repair::floor_int(upper);
        if(ipp_thresholds && channels==1 && height>3 && width>3) {
            // OpenCV's optional IPP branch narrows thresholds to float first.
            // A negative low threshold rejects that branch; all pixel work here
            // remains in this kernel. NaN and huge positive thresholds cannot
            // admit candidates/seeds, unlike cvFloor's overflow sentinel.
            float ipp_low=static_cast<float>(lower),ipp_high=static_cast<float>(upper);
            if(!(ipp_low<0)) {
                low=!(ipp_low<2147483648.0f)?INT_MAX:cn_repair::floor_int(ipp_low);
                high=!(ipp_high<2147483648.0f)?INT_MAX:cn_repair::floor_int(ipp_high);
            }
        }
        size_t pitch=width+2,area=(height+2)*pitch;
        std::vector<int> dx(pixels),dy(pixels),magnitude(area,0);
        std::vector<uint8_t> map(area,1);
        std::vector<size_t> stack;
        Derivatives job{source,height,width,channels,dx.data(),dy.data(),magnitude.data()};
        status=cn_parallel_for(height,8,derivatives,&job);
        if(status!=CN_OK)return status;
        for(size_t y=0;y<height;++y)for(size_t x=0;x<width;++x) {
            size_t index=y*width+x,position=(y+1)*pitch+x+1;
            int m=magnitude[position];
            if(m<=low)continue;
            int gx=dx[index],gy=dy[index];
            int ax=std::abs(gx),ay=std::abs(gy)<<15,tangent=ax*13573;
            bool maximum;
            if(ay<tangent)maximum=m>magnitude[position-1]&&m>=magnitude[position+1];
            else if(ay>tangent+(ax<<16))maximum=m>magnitude[position-pitch]&&m>=magnitude[position+pitch];
            else {
                ptrdiff_t sign=(gx^gy)<0?-1:1;
                auto p=static_cast<ptrdiff_t>(position),step=static_cast<ptrdiff_t>(pitch);
                maximum=m>magnitude[static_cast<size_t>(p-step-sign)]&&m>magnitude[static_cast<size_t>(p+step+sign)];
            }
            if(maximum) {
                map[position]=m>high?2:0;
                if(m>high)stack.push_back(position);
            }
        }
        while(!stack.empty()) {
            size_t p=stack.back();stack.pop_back();
            const size_t neighbors[]={p-pitch-1,p-pitch,p-pitch+1,p-1,p+1,p+pitch-1,p+pitch,p+pitch+1};
            for(size_t q:neighbors)if(map[q]==0){map[q]=2;stack.push_back(q);}
        }
        for(size_t y=0;y<height;++y)for(size_t x=0;x<width;++x)
            output[y*width+x]=map[(y+1)*pitch+x+1]==2?255:0;
        return CN_OK;
    }catch(const std::bad_alloc&){return CN_ALLOCATION_FAILED;}
    catch(const std::length_error&){return CN_SIZE_OVERFLOW;}
    catch(...){return static_cast<cn_status>(4);}
}

extern "C" CN_EXPORT cn_status cn_canny_u8(const uint8_t *source,uint8_t *output,
        size_t height,size_t width,size_t channels,double lower,double upper) noexcept {
    return cn_canny_u8_policy(source,output,height,width,channels,lower,upper,0);
}
