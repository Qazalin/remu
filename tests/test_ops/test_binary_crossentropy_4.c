#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_320n1(float* data0, const float* data1, const float* data2) {
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 320; ridx0++) {
    float val0 = *(data1+ridx0);
    float val1 = *(data2+ridx0);
    float alu0 = (-((val1<0.0f)?0.0f:((0.0f<val1)?val1:(val1*0.5f))));
    float alu1 = max(val0,0.0f);
    float alu2 = max((-val0),0.0f);
    acc0 = ((((val0<0.0f)?0.0f:((0.0f<val0)?val0:(val0*0.5f)))-((-((alu0<(-1.0f))?(-1.0f):(((-1.0f)<alu0)?alu0:((alu0+(-1.0f))*0.5f))))*val0))+(log2((1.0f+exp2(((-(alu1+alu2))*1.4426950408889634f))))*0.6931471805599453f)+acc0);
  }
  *(data0+0) = (acc0*0.003125f);
}