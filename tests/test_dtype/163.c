#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_4_4n16(long* data0, const long* data1) {
  int gidx0 = blockIdx.y; /* 4 */
  int gidx1 = blockIdx.x; /* 4 */
  long val0 = *(data1+gidx1);
  *(data0+(gidx0*4)+gidx1) = (val0+(long)(1));
}