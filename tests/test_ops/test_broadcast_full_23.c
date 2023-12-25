#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_5_8_3_7n3(float* data0, const float* data1, const float* data2, const float* data3, const float* data4) {
  int gidx0 = blockIdx.z; /* 2 */
  int gidx1 = blockIdx.y; /* 5 */
  int gidx2 = blockIdx.x; /* 8 */
  float acc0 = 0.0f;
  float val0 = *(data2+0);
  int alu0 = ((gidx0*40)+(gidx1*8)+gidx2);
  float val1 = *(data4+alu0);
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 7; ridx1++) {
      float val2 = *(data1+(gidx0*840)+(gidx1*56)+gidx2+(ridx0*280)+(ridx1*8));
      float val3 = *(data3+(ridx0*7)+ridx1);
      float alu1 = ((val2+1.0f)*val0);
      acc0 = ((((-(alu1+alu1))*val3)/(val1*val1))+acc0);
    }
  }
  *(data0+alu0) = acc0;
}