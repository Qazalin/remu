#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_10_10_10(float* data0) {
  int gidx0 = blockIdx.y; /* 10 */
  int gidx1 = blockIdx.x; /* 10 */
  float acc0 = 0.0f;
  int alu0 = (gidx0*10);
  for (int ridx0 = 0; ridx0 < 10; ridx0++) {
    acc0 = ((((((alu0+ridx0)%11)<1)?1.0f:0.0f)*((((gidx1+ridx0+2)%11)<1)?1.0f:0.0f))+acc0);
  }
  *(data0+alu0+gidx1) = acc0;
}