#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_8_25_8n2(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.y; /* 8 */
  int gidx1 = blockIdx.x; /* 25 */
  float acc0 = 0.0f;
  int alu0 = ((gidx0*25)+gidx1);
  float val0 = *(data1+alu0);
  for (int ridx0 = 0; ridx0 < 8; ridx0++) {
    float val1 = *(data2+gidx0+(ridx0*8));
    float val2 = *(data3+gidx1+(ridx0*25));
    acc0 = ((val1*val2)+acc0);
  }
  *(data0+alu0) = ((float)((0.0f<val0))*acc0);
}