#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_9_10_6(float* data0, const float* data1) {
  int gidx0 = blockIdx.z; /* 9 */
  int gidx1 = blockIdx.y; /* 10 */
  int gidx2 = blockIdx.x; /* 6 */
  float val0 = ((((gidx2*(-1))<0)*(gidx2<4)*((gidx1*(-1))<(-2))*(gidx1<6)))?(*(data1+(gidx0*9)+(gidx1*3)+gidx2+(-10))):0.0f;
  *(data0+(gidx0*60)+(gidx1*6)+gidx2) = val0;
}