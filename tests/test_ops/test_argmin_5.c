#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_20_10n3(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.x; /* 20 */
  float acc0 = -INFINITY;
  float val0 = *(data2+gidx0);
  for (int ridx0 = 0; ridx0 < 10; ridx0++) {
    float val1 = *(data1+gidx0+(ridx0*20));
    float val2 = *(data3+ridx0);
    float alu0 = (-val1);
    float alu1 = max(((1.0f-(float)(((alu0<val0)+(val0<alu0))))*val2),acc0);
    acc0 = alu1;
  }
  *(data0+gidx0) = (10.0f-acc0-1.0f);
}