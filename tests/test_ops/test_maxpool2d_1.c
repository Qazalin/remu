#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_64_36_9_3_3(float* data0, const float* data1) {
  int gidx0 = blockIdx.z; /* 64 */
  int gidx1 = blockIdx.y; /* 36 */
  int gidx2 = blockIdx.x; /* 9 */
  float acc0 = -INFINITY;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float val0 = *(data1+(gidx0*3080)+(gidx1*84)+(gidx2*3)+(ridx0*28)+ridx1);
      float alu0 = max(val0,acc0);
      acc0 = alu0;
    }
  }
  *(data0+(gidx0*324)+(gidx1*9)+gidx2) = acc0;
}