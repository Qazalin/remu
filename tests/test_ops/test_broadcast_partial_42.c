#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_4_5n5(float* data0, const float* data1, const float* data2, const float* data3, const float* data4) {
  int gidx0 = blockIdx.y; /* 4 */
  int gidx1 = blockIdx.x; /* 5 */
  int alu0 = ((gidx0*5)+gidx1);
  float val0 = *(data1+alu0);
  float val1 = *(data2+0);
  float val2 = *(data3+gidx0);
  float val3 = *(data4+alu0);
  float alu1 = ((val0+1.0f)*val1);
  *(data0+alu0) = (((-(alu1+alu1))*val2)/(val3*val3));
}