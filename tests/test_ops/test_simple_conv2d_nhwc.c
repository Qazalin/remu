#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_20_7_7_10_3_3(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 2 */
  int gidx1 = blockIdx.y; /* 20 */
  int gidx2 = blockIdx.x; /* 49 */
  float acc0 = 0.0f;
  int alu0 = (gidx2/7);
  int alu1 = (gidx2%7);
  for (int ridx0 = 0; ridx0 < 10; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      for (int ridx2 = 0; ridx2 < 3; ridx2++) {
        float val0 = *(data1+(gidx0*810)+(alu0*90)+(alu1*10)+ridx0+(ridx1*90)+(ridx2*10));
        float val1 = *(data2+gidx1+(ridx0*20)+(ridx1*600)+(ridx2*200));
        acc0 = ((val0*val1)+acc0);
      }
    }
  }
  float alu2 = max(acc0,0.0f);
  *(data0+(gidx0*980)+(gidx1*49)+(alu0*7)+alu1) = alu2;
}