#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_256_256(float* data0, const float* data1) {
  int gidx0 = blockIdx.y; /* 4 */
  int gidx1 = blockIdx.x; /* 256 */
  float acc0 = 0.0f;
  int alu0 = (gidx0*256);
  for (int ridx0 = 0; ridx0 < 256; ridx0++) {
    int alu1 = ((gidx1+ridx0+alu0+769)%1024);
    float val0 = (((((gidx1*(-1))+(ridx0*(-1)))<(-254))*((alu1*(-1))<(-1))))?(*(data1+alu1+(-2))):0.0f;
    acc0 = (val0+acc0);
  }
  *(data0+alu0+gidx1) = acc0;
}