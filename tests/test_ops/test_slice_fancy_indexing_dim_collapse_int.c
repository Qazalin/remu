#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_3_3(int* data0) {
  int gidx0 = blockIdx.x; /* 3 */
  int acc0 = 0;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    acc0 = (((((gidx0*(-1))+(ridx0*(-1)))<(-1))?1:0)+acc0);
  }
  *(data0+gidx0) = (acc0+(-1));
}