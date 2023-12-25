#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_312_5_32n2(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.x; /* 312 */
  float acc0 = 0.0f;
  float val0 = *(data3+0);
  for (int ridx0 = 0; ridx0 < 5; ridx0++) {
    for (int ridx1 = 0; ridx1 < 32; ridx1++) {
      int alu0 = ((gidx0*32)+(ridx0*9984)+ridx1);
      float val1 = *(data1+alu0);
      float val2 = *(data2+alu0);
      float alu1 = ((val2+1.0f)*val0);
      acc0 = ((val1*(alu1+alu1))+acc0);
    }
  }
  *(data0+gidx0) = acc0;
}