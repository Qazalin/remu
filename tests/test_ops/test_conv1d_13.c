#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_8_3_2_7_5(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 8 */
  int gidx1 = blockIdx.y; /* 3 */
  int gidx2 = blockIdx.x; /* 14 */
  float acc0 = 0.0f;
  int alu0 = (gidx2%7);
  int alu1 = (gidx2/7);
  for (int ridx0 = 0; ridx0 < 5; ridx0++) {
    float val0 = *(data1+(gidx0*33)+(gidx1*11)+alu0+ridx0);
    float val1 = *(data2+(gidx1*10)+(alu1*5)+ridx0);
    acc0 = ((val0*val1)+acc0);
  }
  float alu2 = max(acc0,0.0f);
  *(data0+(gidx0*42)+(gidx1*14)+(alu1*7)+alu0) = alu2;
}