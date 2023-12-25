#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_1408_5_5_5(float* data0, const float* data1) {
  int gidx0 = blockIdx.y; /* 1408 */
  int gidx1 = blockIdx.x; /* 5 */
  float acc0 = -INFINITY;
  for (int ridx0 = 0; ridx0 < 5; ridx0++) {
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      float val0 = *(data1+(gidx0*140)+(gidx1*5)+(ridx0*28)+ridx1);
      float alu0 = max(val0,acc0);
      acc0 = alu0;
    }
  }
  *(data0+(gidx0*5)+gidx1) = acc0;
}