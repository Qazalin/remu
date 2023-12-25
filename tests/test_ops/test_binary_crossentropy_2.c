#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_320(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.x; /* 320 */
  float val0 = *(data1+gidx0);
  float val1 = *(data2+gidx0);
  float val2 = *(data3+0);
  float alu0 = (1.0f/(1.0f+exp2((val0*(-1.4426950408889634f)))));
  float alu1 = (1.0f-alu0);
  float alu2 = (-((val1<0.0f)?0.0f:((0.0f<val1)?val1:(val1*0.5f))));
  float alu3 = (-((alu2<(-1.0f))?(-1.0f):(((-1.0f)<alu2)?alu2:((alu2+(-1.0f))*0.5f))));
  *(data0+gidx0) = (alu0*alu1*((-(((1.0f-alu3)*(-val2))/alu1))+(((-alu3)*val2)/alu0)));
}