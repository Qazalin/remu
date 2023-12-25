#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_5_312_32n1(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 5 */
  int gidx1 = blockIdx.y; /* 312 */
  int gidx2 = blockIdx.x; /* 32 */
  int alu0 = ((gidx0*9984)+(gidx1*32)+gidx2);
  float val0 = *(data1+alu0);
  float val1 = *(data2+gidx1);
  *(data0+alu0) = (val0-val1);
}