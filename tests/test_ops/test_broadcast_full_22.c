#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_3_7_2_5_8n2(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.y; /* 3 */
  int gidx1 = blockIdx.x; /* 7 */
  float acc0 = 0.0f;
  float val0 = *(data2+0);
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      for (int ridx2 = 0; ridx2 < 8; ridx2++) {
        float val1 = *(data1+(gidx0*280)+(gidx1*8)+(ridx0*840)+(ridx1*56)+ridx2);
        float val2 = *(data3+(ridx0*40)+(ridx1*8)+ridx2);
        float alu0 = ((val1+1.0f)*val0);
        acc0 = (((alu0+alu0)/val2)+acc0);
      }
    }
  }
  *(data0+(gidx0*7)+gidx1) = acc0;
}