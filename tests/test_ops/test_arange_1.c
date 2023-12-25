#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_2(int* data0) {
  int gidx0 = blockIdx.x; /* 2 */
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    acc0 = (((((gidx0*(-1))+(ridx0*(-1)))<0)?3:0)+acc0);
  }
  *(data0+gidx0) = (acc0+2);
}