#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_8775_3(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.y; /* 8775 */
  int gidx1 = blockIdx.x; /* 3 */
  float val0 = ((gidx1<1))?(*(data1+gidx0)):0.0f;
  int alu0 = (gidx1*(-1));
  float val1 = (((alu0<0)*(gidx1<2)))?(*(data2+gidx0)):0.0f;
  float val2 = ((alu0<(-1)))?(*(data3+gidx0)):0.0f;
  *(data0+(gidx0*3)+gidx1) = (val0+val1+val2);
}