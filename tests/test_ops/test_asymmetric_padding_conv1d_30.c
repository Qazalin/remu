#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_5_2n1(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.x; /* 5 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    float val0 = ((((gidx0*(-1))+(ridx0*(-1)))<(-1)))?(*(data1+gidx0+ridx0+(-2))):0.0f;
    float val1 = *(data2+ridx0);
    acc0 = ((val0*val1)+acc0);
  }
  float alu0 = max(acc0,0.0f);
  *(data0+gidx0) = alu0;
}