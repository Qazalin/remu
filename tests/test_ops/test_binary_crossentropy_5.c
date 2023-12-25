#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_320n2(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.x; /* 320 */
  float val0 = *(data1+gidx0);
  float val1 = *(data2+0);
  float val2 = *(data3+gidx0);
  float alu0 = max((-val0),0.0f);
  float alu1 = max(val0,0.0f);
  float alu2 = exp2(((-(alu1+alu0))*1.4426950408889634f));
  float alu3 = (-(alu2*(val1/(1.0f+alu2))));
  float alu4 = (-((val2<0.0f)?0.0f:((0.0f<val2)?val2:(val2*0.5f))));
  bool alu5 = (0.0f<val0);
  float alu6 = ((val0<0.0f)?0.0f:val1);
  *(data0+gidx0) = ((-((float)((0.0f<alu0))*alu3))+((float)((0.0f<alu1))*alu3)+((-((alu4<(-1.0f))?(-1.0f):(((-1.0f)<alu4)?alu4:((alu4+(-1.0f))*0.5f))))*(-val1))+(alu5?alu6:0.0f)+(0.5f*(alu5?0.0f:alu6)));
}