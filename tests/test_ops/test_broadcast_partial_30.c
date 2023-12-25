#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_4_5n3(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.y; /* 4 */
  int gidx1 = blockIdx.x; /* 5 */
  float val0 = *(data1+gidx0);
  int alu0 = ((gidx0*5)+gidx1);
  float val1 = *(data2+alu0);
  float val2 = *(data3+0);
  float alu1 = ((val1+1.0f)*val2);
  *(data0+alu0) = (val0*(alu1+alu1));
}