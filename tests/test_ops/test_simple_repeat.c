#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_3_9_12(float* data0, const float* data1) {
  int gidx0 = blockIdx.z; /* 3 */
  int gidx1 = blockIdx.y; /* 9 */
  int gidx2 = blockIdx.x; /* 12 */
  float val0 = *(data1+(gidx2%3)+((gidx1%3)*3));
  *(data0+(gidx0*108)+(gidx1*12)+gidx2) = val0;
}