#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_8_24_9(float* data0, const float* data1) {
  int gidx0 = blockIdx.z; /* 8 */
  int gidx1 = blockIdx.y; /* 24 */
  int gidx2 = blockIdx.x; /* 9 */
  float val0 = *(data1+(gidx2%3)+((gidx1%6)*3)+((gidx0%4)*18));
  *(data0+(gidx0*216)+(gidx1*9)+gidx2) = val0;
}