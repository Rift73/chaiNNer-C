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
// Third party copyrights are property of their respective icvers.
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
/* Adapted OpenCV 5.0.0 modules/photo/src/inpaint.cpp; independent C++ buffers,
 * stable heap and morphological setup replace the OpenCV object/runtime layer.
 * Arithmetic expressions and traversal order are preserved. Tiny-image invalid
 * source reads are explicitly clamped, correcting the original undefined reads.
 */
#include "repair_buffers.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <queue>
#include <stdexcept>
#include <vector>

namespace {
using uchar=uint8_t;
using std::abs;
using std::fabs;
using std::sqrt;
struct Point2f { float x,y; };
struct CvMat {
    int rows,cols,type;
    void *data;
    bool clamp_pixels=false;
    template<class T> T &element(int row,int column) const noexcept {
        if(clamp_pixels) {
            row=std::clamp(row,0,rows-1);
            int pixel=column>=0?column/type:-((-column+type-1)/type);
            int component=column-pixel*type;
            column=std::clamp(pixel,0,cols-1)*type+component;
        }
        return static_cast<T *>(data)[static_cast<ptrdiff_t>(row)*cols*type+column];
    }
};
#define CV_MAT_ELEM(mat,type,row,col) ((mat).element<type>((row),(col)))
#define CV_MAT_CN(type) (type)
#define MIN(a,b) ((a)<(b)?(a):(b))

template<class T,class U> T saturate_cast(U value) noexcept {
    int rounded=cn_repair::round_int(static_cast<double>(value));
    return static_cast<T>(std::clamp(rounded,0,255));
}
// OpenCV 5.0's integer round_cast: the half is added in double, then cvRound.
template<class T> T round_cast(float value) noexcept {
    return saturate_cast<T>(value+0.5);
}
class CvPriorityQueueFloat {
    struct Item { float time; uint64_t order; int row,column; };
    struct Compare {
        bool operator()(const Item &a,const Item &b) const noexcept {
            return a.time>b.time || (a.time==b.time && a.order>b.order);
        }
    };
    std::priority_queue<Item,std::vector<Item>,Compare> queue;
    uint64_t order=0;
public:
    void Push(int row,int column,float time) {queue.push({time,order++,row,column});}
    bool Pop(int *row,int *column) {
        if(queue.empty())return false;
        *row=queue.top().row;*column=queue.top().column;queue.pop();return true;
    }
    void Add(const CvMat &mask) {
        for(int r=0;r<mask.rows;++r)for(int c=0;c<mask.cols;++c)
            if(mask.element<uchar>(r,c))Push(r,c,0);
    }
};
inline float
min4( float a, float b, float c, float d )
{
    a = MIN(a,b);
    c = MIN(c,d);
    return MIN(a,c);
}

#define CV_MAT_3COLOR_ELEM(img,type,y,x,c) CV_MAT_ELEM(img,type,y,(x)*3+(c))
#define KNOWN  0  //known outside narrow band
#define BAND   1  //narrow band (known)
#define INSIDE 2  //unknown
#define CHANGE 3  //servise

static inline float VectorScalMult(const Point2f& v1, const Point2f& v2)
{
   return v1.x*v2.x+v1.y*v2.y;
}

static inline float VectorLength(const Point2f& v1)
{
    return v1.x*v1.x+v1.y*v1.y;
}

///////////////////////////////////////////////////////////////////////////////////////////
//HEAP::iterator Heap_Iterator;
//HEAP Heap;

static float FastMarching_solve(int i1,int j1,int i2,int j2, const CvMat* f, const CvMat* t)
{
    double sol, a11, a22, m12;
    a11=CV_MAT_ELEM(*t,float,i1,j1);
    a22=CV_MAT_ELEM(*t,float,i2,j2);
    m12=MIN(a11,a22);

    if( CV_MAT_ELEM(*f,uchar,i1,j1) != INSIDE )
        if( CV_MAT_ELEM(*f,uchar,i2,j2) != INSIDE )
            if( fabs(a11-a22) >= 1.0 )
                sol = 1+m12;
            else
                sol = (a11+a22+sqrt((double)(2-(a11-a22)*(a11-a22))))*0.5;
        else
            sol = 1+a11;
    else if( CV_MAT_ELEM(*f,uchar,i2,j2) != INSIDE )
        sol = 1+a22;
    else
        sol = 1+m12;

    return (float)sol;
}

/////////////////////////////////////////////////////////////////////////////////////


static void
icvCalcFMM(const CvMat *f, CvMat *t, CvPriorityQueueFloat *Heap, bool negate) {
   int i, j, ii = 0, jj = 0, q;
   float dist;

   while (Heap->Pop(&ii,&jj)) {

      unsigned known=(negate)?CHANGE:KNOWN;
      CV_MAT_ELEM(*f,uchar,ii,jj) = (uchar)known;

      for (q=0; q<4; q++) {
         i=0; j=0;
         if     (q==0) {i=ii-1; j=jj;}
         else if(q==1) {i=ii;   j=jj-1;}
         else if(q==2) {i=ii+1; j=jj;}
         else {i=ii;   j=jj+1;}
         if ((i<=0)||(j<=0)||(i>f->rows)||(j>f->cols)) continue;

         if (CV_MAT_ELEM(*f,uchar,i,j)==INSIDE) {
            dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                        FastMarching_solve(i+1,j,i,j-1,f,t),
                        FastMarching_solve(i-1,j,i,j+1,f,t),
                        FastMarching_solve(i+1,j,i,j+1,f,t));
            CV_MAT_ELEM(*t,float,i,j) = dist;
            CV_MAT_ELEM(*f,uchar,i,j) = BAND;
            Heap->Push(i,j,dist);
         }
      }
   }

   if (negate) {
      for (i=0; i<f->rows; i++) {
         for(j=0; j<f->cols; j++) {
            if (CV_MAT_ELEM(*f,uchar,i,j) == CHANGE) {
               CV_MAT_ELEM(*f,uchar,i,j) = KNOWN;
               CV_MAT_ELEM(*t,float,i,j) = -CV_MAT_ELEM(*t,float,i,j);
            }
         }
      }
   }
}

template <typename data_type>
static void
icvTeleaInpaintFMM(const CvMat *f, CvMat *t, CvMat *out, int range, CvPriorityQueueFloat *Heap ) {
   int i = 0, j = 0, ii = 0, jj = 0, k, l, q, color = 0;
   float dist;

   if (CV_MAT_CN(out->type)==3) {

      while (Heap->Pop(&ii,&jj)) {

         CV_MAT_ELEM(*f,uchar,ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t->rows-1)||(j>t->cols-1)) continue;

            if (CV_MAT_ELEM(*f,uchar,i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               CV_MAT_ELEM(*t,float,i,j) = dist;

               Point2f gradT[3];
               for (color=0; color<=2; color++) {
                  if (CV_MAT_ELEM(*f,uchar,i,j+1)!=INSIDE) {
                     if (CV_MAT_ELEM(*f,uchar,i,j-1)!=INSIDE) {
                        gradT[color].x=(float)((CV_MAT_ELEM(*t,float,i,j+1)-CV_MAT_ELEM(*t,float,i,j-1)))*0.5f;
                     } else {
                        gradT[color].x=(float)((CV_MAT_ELEM(*t,float,i,j+1)-CV_MAT_ELEM(*t,float,i,j)));
                     }
                  } else {
                     if (CV_MAT_ELEM(*f,uchar,i,j-1)!=INSIDE) {
                        gradT[color].x=(float)((CV_MAT_ELEM(*t,float,i,j)-CV_MAT_ELEM(*t,float,i,j-1)));
                     } else {
                        gradT[color].x=0;
                     }
                  }
                  if (CV_MAT_ELEM(*f,uchar,i+1,j)!=INSIDE) {
                     if (CV_MAT_ELEM(*f,uchar,i-1,j)!=INSIDE) {
                        gradT[color].y=(float)((CV_MAT_ELEM(*t,float,i+1,j)-CV_MAT_ELEM(*t,float,i-1,j)))*0.5f;
                     } else {
                        gradT[color].y=(float)((CV_MAT_ELEM(*t,float,i+1,j)-CV_MAT_ELEM(*t,float,i,j)));
                     }
                  } else {
                     if (CV_MAT_ELEM(*f,uchar,i-1,j)!=INSIDE) {
                        gradT[color].y=(float)((CV_MAT_ELEM(*t,float,i,j)-CV_MAT_ELEM(*t,float,i-1,j)));
                     } else {
                        gradT[color].y=0;
                     }
                  }
               }

               Point2f gradI,r;
               float Jx[3] = {0,0,0};
               float Jy[3] = {0,0,0};
               float Ia[3] = {0,0,0};
               float s[3] = {1.0e-20f,1.0e-20f,1.0e-20f};
               float w,dst,lev,dir,sat;

               for (k=i-range; k<=i+range; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==t->rows-2);
                  for (l=j-range; l<=j+range; l++) {
                     int lm=l-1+(l==1),lp=l-1-(l==t->cols-2);
                     if (k>0&&l>0&&k<t->rows-1&&l<t->cols-1) {
                        if ((CV_MAT_ELEM(*f,uchar,k,l)!=INSIDE)&&
                            ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                           for (color=0; color<=2; color++) {
                              r.y     = (float)(i-k);
                              r.x     = (float)(j-l);

                              dst = (float)(1./(VectorLength(r)*sqrt((double)VectorLength(r))));
                              lev = (float)(1./(1+fabs(CV_MAT_ELEM(*t,float,k,l)-CV_MAT_ELEM(*t,float,i,j))));

                              dir=VectorScalMult(r,gradT[color]);
                              if (fabs(dir)<=0.01) dir=0.000001f;
                              w = (float)fabs(dst*lev*dir);

                              if (CV_MAT_ELEM(*f,uchar,k,l+1)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.x=(float)((CV_MAT_3COLOR_ELEM(*out,uchar,km,lp+1,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm-1,color)))*2.0f;
                                 } else {
                                    gradI.x=(float)((CV_MAT_3COLOR_ELEM(*out,uchar,km,lp+1,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm,color)));
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.x=(float)((CV_MAT_3COLOR_ELEM(*out,uchar,km,lp,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm-1,color)));
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (CV_MAT_ELEM(*f,uchar,k+1,l)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.y=(float)((CV_MAT_3COLOR_ELEM(*out,uchar,kp+1,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km-1,lm,color)))*2.0f;
                                 } else {
                                    gradI.y=(float)((CV_MAT_3COLOR_ELEM(*out,uchar,kp+1,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm,color)));
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.y=(float)((CV_MAT_3COLOR_ELEM(*out,uchar,kp,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km-1,lm,color)));
                                 } else {
                                    gradI.y=0;
                                 }
                              }
                              Ia[color] += (float)w * (float)(CV_MAT_3COLOR_ELEM(*out,uchar,k-1,l-1,color));
                              Jx[color] -= (float)w * (float)(gradI.x*r.x);
                              Jy[color] -= (float)w * (float)(gradI.y*r.y);
                              s[color]  += w;
                           }
                        }
                     }
                  }
               }
               for (color=0; color<=2; color++) {
                  sat = (float)(Ia[color]/s[color]+(Jx[color]+Jy[color])/(sqrt(Jx[color]*Jx[color]+Jy[color]*Jy[color])+1.0e-20f));
                  CV_MAT_3COLOR_ELEM(*out,uchar,i-1,j-1,color) = round_cast<uchar>(sat);
               }

               CV_MAT_ELEM(*f,uchar,i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }

   } else if (CV_MAT_CN(out->type)==1) {

      while (Heap->Pop(&ii,&jj)) {

         CV_MAT_ELEM(*f,uchar,ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t->rows-1)||(j>t->cols-1)) continue;

            if (CV_MAT_ELEM(*f,uchar,i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               CV_MAT_ELEM(*t,float,i,j) = dist;

               for (color=0; color<=0; color++) {
                  Point2f gradI,gradT,r;
                  float Ia=0,Jx=0,Jy=0,s=1.0e-20f,w,dst,lev,dir,sat;

                  if (CV_MAT_ELEM(*f,uchar,i,j+1)!=INSIDE) {
                     if (CV_MAT_ELEM(*f,uchar,i,j-1)!=INSIDE) {
                        gradT.x=(float)((CV_MAT_ELEM(*t,float,i,j+1)-CV_MAT_ELEM(*t,float,i,j-1)))*0.5f;
                     } else {
                        gradT.x=(float)((CV_MAT_ELEM(*t,float,i,j+1)-CV_MAT_ELEM(*t,float,i,j)));
                     }
                  } else {
                     if (CV_MAT_ELEM(*f,uchar,i,j-1)!=INSIDE) {
                        gradT.x=(float)((CV_MAT_ELEM(*t,float,i,j)-CV_MAT_ELEM(*t,float,i,j-1)));
                     } else {
                        gradT.x=0;
                     }
                  }
                  if (CV_MAT_ELEM(*f,uchar,i+1,j)!=INSIDE) {
                     if (CV_MAT_ELEM(*f,uchar,i-1,j)!=INSIDE) {
                        gradT.y=(float)((CV_MAT_ELEM(*t,float,i+1,j)-CV_MAT_ELEM(*t,float,i-1,j)))*0.5f;
                     } else {
                        gradT.y=(float)((CV_MAT_ELEM(*t,float,i+1,j)-CV_MAT_ELEM(*t,float,i,j)));
                     }
                  } else {
                     if (CV_MAT_ELEM(*f,uchar,i-1,j)!=INSIDE) {
                        gradT.y=(float)((CV_MAT_ELEM(*t,float,i,j)-CV_MAT_ELEM(*t,float,i-1,j)));
                     } else {
                        gradT.y=0;
                     }
                  }
                  for (k=i-range; k<=i+range; k++) {
                     int km=k-1+(k==1),kp=k-1-(k==t->rows-2);
                     for (l=j-range; l<=j+range; l++) {
                        int lm=l-1+(l==1),lp=l-1-(l==t->cols-2);
                        if (k>0&&l>0&&k<t->rows-1&&l<t->cols-1) {
                           if ((CV_MAT_ELEM(*f,uchar,k,l)!=INSIDE)&&
                               ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                              r.y     = (float)(i-k);
                              r.x     = (float)(j-l);

                              dst = (float)(1./(VectorLength(r)*sqrt(VectorLength(r))));
                              lev = (float)(1./(1+fabs(CV_MAT_ELEM(*t,float,k,l)-CV_MAT_ELEM(*t,float,i,j))));

                              dir=VectorScalMult(r,gradT);
                              if (fabs(dir)<=0.01) dir=0.000001f;
                              w = (float)fabs(dst*lev*dir);

                              if (CV_MAT_ELEM(*f,uchar,k,l+1)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.x=(float)((CV_MAT_ELEM(*out,data_type,km,lp+1)-CV_MAT_ELEM(*out,data_type,km,lm-1)))*2.0f;
                                 } else {
                                    gradI.x=(float)((CV_MAT_ELEM(*out,data_type,km,lp+1)-CV_MAT_ELEM(*out,data_type,km,lm)));
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.x=(float)((CV_MAT_ELEM(*out,data_type,km,lp)-CV_MAT_ELEM(*out,data_type,km,lm-1)));
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (CV_MAT_ELEM(*f,uchar,k+1,l)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.y=(float)((CV_MAT_ELEM(*out,data_type,kp+1,lm)-CV_MAT_ELEM(*out,data_type,km-1,lm)))*2.0f;
                                 } else {
                                    gradI.y=(float)((CV_MAT_ELEM(*out,data_type,kp+1,lm)-CV_MAT_ELEM(*out,data_type,km,lm)));
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.y=(float)((CV_MAT_ELEM(*out,data_type,kp,lm)-CV_MAT_ELEM(*out,data_type,km-1,lm)));
                                 } else {
                                    gradI.y=0;
                                 }
                              }
                              Ia += (float)w * (float)(CV_MAT_ELEM(*out,data_type,k-1,l-1));
                              Jx -= (float)w * (float)(gradI.x*r.x);
                              Jy -= (float)w * (float)(gradI.y*r.y);
                              s  += w;
                           }
                        }
                     }
                  }
                  sat = (float)(Ia/s+(Jx+Jy)/(sqrt(Jx*Jx+Jy*Jy)+1.0e-20f));
                  {
                  CV_MAT_ELEM(*out,data_type,i-1,j-1) = round_cast<data_type>(sat);
                  }
               }

               CV_MAT_ELEM(*f,uchar,i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }
   }
}

template <typename data_type>
static void
icvNSInpaintFMM(const CvMat *f, CvMat *t, CvMat *out, int range, CvPriorityQueueFloat *Heap) {
   int i = 0, j = 0, ii = 0, jj = 0, k, l, q, color = 0;
   float dist;

   if (CV_MAT_CN(out->type)==3) {

      while (Heap->Pop(&ii,&jj)) {

         CV_MAT_ELEM(*f,uchar,ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t->rows-1)||(j>t->cols-1)) continue;

            if (CV_MAT_ELEM(*f,uchar,i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               CV_MAT_ELEM(*t,float,i,j) = dist;

               Point2f gradI,r;
               float Ia[3]={0,0,0};
               float s[3]={1.0e-20f,1.0e-20f,1.0e-20f};
               float w,dst,dir;

               for (k=i-range; k<=i+range; k++) {
                  int km=k-1+(k==1),kp=k-1-(k==f->rows-2);
                  for (l=j-range; l<=j+range; l++) {
                     int lm=l-1+(l==1),lp=l-1-(l==f->cols-2);
                     if (k>0&&l>0&&k<f->rows-1&&l<f->cols-1) {
                        if ((CV_MAT_ELEM(*f,uchar,k,l)!=INSIDE)&&
                            ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                           for (color=0; color<=2; color++) {
                              r.y=(float)(k-i);
                              r.x=(float)(l-j);

                              dst = 1/(VectorLength(r)*VectorLength(r)+1);

                              if (CV_MAT_ELEM(*f,uchar,k+1,l)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.x=(float)(abs(CV_MAT_3COLOR_ELEM(*out,uchar,kp+1,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,kp,lm,color))+
                                                    abs(CV_MAT_3COLOR_ELEM(*out,uchar,kp,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km-1,lm,color)));
                                 } else {
                                    gradI.x=(float)(abs(CV_MAT_3COLOR_ELEM(*out,uchar,kp+1,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,kp,lm,color)))*2.0f;
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.x=(float)(abs(CV_MAT_3COLOR_ELEM(*out,uchar,kp,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km-1,lm,color)))*2.0f;
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (CV_MAT_ELEM(*f,uchar,k,l+1)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.y=(float)(abs(CV_MAT_3COLOR_ELEM(*out,uchar,km,lp+1,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm,color))+
                                                    abs(CV_MAT_3COLOR_ELEM(*out,uchar,km,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm-1,color)));
                                 } else {
                                    gradI.y=(float)(abs(CV_MAT_3COLOR_ELEM(*out,uchar,km,lp+1,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm,color)))*2.0f;
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.y=(float)(abs(CV_MAT_3COLOR_ELEM(*out,uchar,km,lm,color)-CV_MAT_3COLOR_ELEM(*out,uchar,km,lm-1,color)))*2.0f;
                                 } else {
                                    gradI.y=0;
                                 }
                              }

                              gradI.x=-gradI.x;
                              dir=VectorScalMult(r,gradI);

                              if (fabs(dir)<=0.01) {
                                 dir=0.000001f;
                              } else {
                                 dir = (float)fabs(VectorScalMult(r,gradI)/sqrt(VectorLength(r)*VectorLength(gradI)));
                              }
                              w = dst*dir;
                              Ia[color] += (float)w * (float)(CV_MAT_3COLOR_ELEM(*out,uchar,k-1,l-1,color));
                              s[color]  += w;
                           }
                        }
                     }
                  }
               }
               for (color=0; color<=2; color++) {
                  CV_MAT_3COLOR_ELEM(*out,uchar,i-1,j-1,color) = saturate_cast<uchar>((double)Ia[color]/s[color]);
               }

               CV_MAT_ELEM(*f,uchar,i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }

   } else if (CV_MAT_CN(out->type)==1) {

      while (Heap->Pop(&ii,&jj)) {

         CV_MAT_ELEM(*f,uchar,ii,jj) = KNOWN;
         for(q=0; q<4; q++) {
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>t->rows-1)||(j>t->cols-1)) continue;

            if (CV_MAT_ELEM(*f,uchar,i,j)==INSIDE) {
               dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
               CV_MAT_ELEM(*t,float,i,j) = dist;

               {
                  Point2f gradI,r;
                  float Ia=0,s=1.0e-20f,w,dst,dir;

                  for (k=i-range; k<=i+range; k++) {
                     int km=k-1+(k==1),kp=k-1-(k==t->rows-2);
                     for (l=j-range; l<=j+range; l++) {
                        int lm=l-1+(l==1),lp=l-1-(l==t->cols-2);
                        if (k>0&&l>0&&k<t->rows-1&&l<t->cols-1) {
                           if ((CV_MAT_ELEM(*f,uchar,k,l)!=INSIDE)&&
                               ((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                              r.y=(float)(i-k);
                              r.x=(float)(j-l);

                              dst = 1/(VectorLength(r)*VectorLength(r)+1);

                              if (CV_MAT_ELEM(*f,uchar,k+1,l)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.x=(float)(std::abs(CV_MAT_ELEM(*out,data_type,kp+1,lm)-CV_MAT_ELEM(*out,data_type,kp,lm))+
                                                    std::abs(CV_MAT_ELEM(*out,data_type,kp,lm)-CV_MAT_ELEM(*out,data_type,km-1,lm)));
                                 } else {
                                    gradI.x=(float)(std::abs(CV_MAT_ELEM(*out,data_type,kp+1,lm)-CV_MAT_ELEM(*out,data_type,kp,lm)))*2.0f;
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                    gradI.x=(float)(std::abs(CV_MAT_ELEM(*out,data_type,kp,lm)-CV_MAT_ELEM(*out,data_type,km-1,lm)))*2.0f;
                                 } else {
                                    gradI.x=0;
                                 }
                              }
                              if (CV_MAT_ELEM(*f,uchar,k,l+1)!=INSIDE) {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.y=(float)(std::abs(CV_MAT_ELEM(*out,data_type,km,lp+1)-CV_MAT_ELEM(*out,data_type,km,lm))+
                                                    std::abs(CV_MAT_ELEM(*out,data_type,km,lm)-CV_MAT_ELEM(*out,data_type,km,lm-1)));
                                 } else {
                                    gradI.y=(float)(std::abs(CV_MAT_ELEM(*out,data_type,km,lp+1)-CV_MAT_ELEM(*out,data_type,km,lm)))*2.0f;
                                 }
                              } else {
                                 if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                    gradI.y=(float)(std::abs(CV_MAT_ELEM(*out,data_type,km,lm)-CV_MAT_ELEM(*out,data_type,km,lm-1)))*2.0f;
                                 } else {
                                    gradI.y=0;
                                 }
                              }

                              gradI.x=-gradI.x;
                              dir=VectorScalMult(r,gradI);

                              if (fabs(dir)<=0.01) {
                                 dir=0.000001f;
                              } else {
                                 dir = (float)fabs(VectorScalMult(r,gradI)/sqrt(VectorLength(r)*VectorLength(gradI)));
                              }
                              w = dst*dir;
                              Ia += (float)w * (float)(CV_MAT_ELEM(*out,data_type,k-1,l-1));
                              s  += w;
                           }
                        }
                     }
                  }
                  CV_MAT_ELEM(*out,data_type,i-1,j-1) = saturate_cast<data_type>((double)Ia/s);
               }

               CV_MAT_ELEM(*f,uchar,i,j) = BAND;
               Heap->Push(i,j,dist);
            }
         }
      }

   }
}


void clear_border(std::vector<uchar> &data,int height,int width) {
    std::fill_n(data.begin(),width,uchar(0));
    std::fill_n(data.end()-width,width,uchar(0));
    for(int y=1;y<height-1;++y){data[static_cast<size_t>(y)*width]=0;data[static_cast<size_t>(y+1)*width-1]=0;}
}
// A separable sliding occupancy count implements rectangular dilation exactly.
std::vector<uchar> dilate_rectangle(const std::vector<uchar> &mask,int height,int width,int radius) {
    std::vector<uchar> horizontal(mask.size()),out(mask.size());
    for(int y=0;y<height;++y) {
        size_t row=static_cast<size_t>(y)*width;
        int count=0;
        for(int x=0;x<=std::min(radius,width-1);++x)count+=mask[row+x]!=0;
        for(int x=0;x<width;++x) {
            horizontal[row+x]=count?2:0;
            if(x-radius>=0)count-=mask[row+static_cast<size_t>(x-radius)]!=0;
            if(x+radius+1<width)count+=mask[row+static_cast<size_t>(x+radius+1)]!=0;
        }
    }
    for(int x=0;x<width;++x) {
        int count=0;
        for(int y=0;y<=std::min(radius,height-1);++y)count+=horizontal[static_cast<size_t>(y)*width+x]!=0;
        for(int y=0;y<height;++y) {
            out[static_cast<size_t>(y)*width+x]=count?2:0;
            if(y-radius>=0)count-=horizontal[static_cast<size_t>(y-radius)*width+x]!=0;
            if(y+radius+1<height)count+=horizontal[static_cast<size_t>(y+radius+1)*width+x]!=0;
        }
    }
    return out;
}
}

extern "C" CN_EXPORT cn_status cn_inpaint_u8(const uint8_t *source,const uint8_t *input_mask,
        uint8_t *output,size_t height,size_t width,size_t channels,double radius,int method) noexcept {
    size_t pixels;
    cn_status status=cn_repair::dimensions(height,width,channels,pixels);
    if(status!=CN_OK)return status;
    if((channels!=1&&channels!=3)||(method!=0&&method!=1))return CN_INVALID_ARGUMENT;
    cn_repair::Region src{},mask_region{},out{};
    status=cn_repair::region(source,pixels*channels,src);if(status!=CN_OK)return status;
    status=cn_repair::region(input_mask,pixels,mask_region);if(status!=CN_OK)return status;
    status=cn_repair::region(output,pixels*channels,out);if(status!=CN_OK)return status;
    if(cn_repair::overlap(src,out)||cn_repair::overlap(mask_region,out))return CN_INVALID_ARGUMENT;
    try {
        int h=static_cast<int>(height)+2,w=static_cast<int>(width)+2;
        size_t area=static_cast<size_t>(h)*w;
        int range=std::clamp(cn_repair::round_int(radius),1,100);
        std::vector<uchar> mask(area,0),band(area,0),result(source,source+pixels*channels);
        std::vector<float> times(area,1.0e6f);
        for(size_t y=0;y<height;++y)for(size_t x=0;x<width;++x)
            if(input_mask[y*width+x])mask[(y+1)*static_cast<size_t>(w)+x+1]=INSIDE;
        // The cross dilation minus the mask is the initial known front.
        for(int y=1;y<h-1;++y)for(int x=1;x<w-1;++x) {
            size_t p=static_cast<size_t>(y)*w+x;
            if(!mask[p]&&(mask[p-w]||mask[p-1]||mask[p+1]||mask[p+w]))band[p]=2;
        }
        for(size_t p=0;p<area;++p)if(band[p])times[p]=0;
        CvMat mask_mat{h,w,1,mask.data()},band_mat{h,w,1,band.data()},time_mat{h,w,1,times.data()};
        CvMat result_mat{static_cast<int>(height),static_cast<int>(width),static_cast<int>(channels),result.data(),height==1||width==1};
        CvPriorityQueueFloat heap;
        heap.Add(band_mat);
        if(method==1) {
            auto outside=dilate_rectangle(mask,h,w,range);
            for(size_t p=0;p<area;++p)outside[p]=static_cast<uchar>(std::max(0,static_cast<int>(outside[p])-mask[p]));
            CvPriorityQueueFloat outer;
            outer.Add(band_mat);
            for(size_t p=0;p<area;++p)outside[p]=static_cast<uchar>(std::max(0,static_cast<int>(outside[p])-band[p]));
            clear_border(outside,h,w);
            CvMat outside_mat{h,w,1,outside.data()};
            icvCalcFMM(&outside_mat,&time_mat,&outer,true);
            icvTeleaInpaintFMM<uchar>(&mask_mat,&time_mat,&result_mat,range,&heap);
        } else icvNSInpaintFMM<uchar>(&mask_mat,&time_mat,&result_mat,range,&heap);
        std::memcpy(output,result.data(),pixels*channels);
        return CN_OK;
    }catch(const std::bad_alloc&){return CN_ALLOCATION_FAILED;}
    catch(const std::length_error&){return CN_SIZE_OVERFLOW;}
    catch(...){return static_cast<cn_status>(4);}
}
