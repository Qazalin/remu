#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_2_5_5(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.y; /* 2 */
  int gidx1 = blockIdx.x; /* 2 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 5; ridx0++) {
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      int alu0 = (gidx1+ridx1);
      float val0 = (((((gidx1*(-1))+(ridx1*(-1)))<(-1))*(alu0<5)*(((gidx0*(-1))+(ridx0*(-1)))<(-1))*((gidx0+ridx0)<5)))?(*(data1+alu0+(gidx0*3)+(ridx0*3)+(-8))):0.0f;
      float val1 = *(data2+(ridx0*5)+ridx1);
      acc0 = ((val0*val1)+acc0);
    }
  }
  *(data0+(gidx0*2)+gidx1) = acc0;
}