#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_64_111_28(float* data0, const float* data1) {
  int gidx0 = blockIdx.z; /* 64 */
  int gidx1 = blockIdx.y; /* 111 */
  int gidx2 = blockIdx.x; /* 28 */
  float val0 = *(data1+(gidx2/2)+(gidx0*518)+((gidx1/3)*14));
  *(data0+(gidx0*3108)+(gidx1*28)+gidx2) = val0;
}