#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_3_4(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.y; /* 3 */
  int gidx1 = blockIdx.x; /* 4 */
  float val0 = ((((gidx1*(-1))<0)*(gidx1<3)*((gidx0*(-1))<0)*(gidx0<2)))?(*(data1+gidx1+(-1))):0.0f;
  float val1 = *(data2+0);
  float alu0 = max((val0*val1),0.0f);
  *(data0+(gidx0*4)+gidx1) = alu0;
}