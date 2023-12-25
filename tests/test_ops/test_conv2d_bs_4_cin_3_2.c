#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_6_11_5_3_3(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 4 */
  int gidx1 = blockIdx.y; /* 6 */
  int gidx2 = blockIdx.x; /* 55 */
  float acc0 = 0.0f;
  int alu0 = (gidx2/5);
  int alu1 = (gidx2%5);
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      float val0 = *(data1+(gidx0*231)+(alu0*7)+alu1+(ridx0*77)+ridx1);
      float val1 = *(data2+(gidx1*9)+(ridx0*3)+ridx1);
      acc0 = ((val0*val1)+acc0);
    }
  }
  float alu2 = max(acc0,0.0f);
  *(data0+(gidx0*330)+(gidx1*55)+(alu0*5)+alu1) = alu2;
}