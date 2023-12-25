#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_2925n15(float* data0, const float* data1) {
  int gidx0 = blockIdx.x; /* 2925 */
  float val0 = *(data1+gidx0);
  *(data0+gidx0) = (0.5f*val0*(1.0f+((2.0f*(1.0f/(1.0f+exp2((2.0f*(val0*0.7978845608f*(1.0f+(0.044715f*val0*val0)))*(-1.4426950408889634f))))))-1.0f)));
}