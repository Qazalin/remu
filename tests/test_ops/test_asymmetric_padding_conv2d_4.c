#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_4_3_3(float* data0, const float* data1) {
  int gidx0 = blockIdx.y; /* 4 */
  int gidx1 = blockIdx.x; /* 4 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    int alu0 = (gidx0+(ridx0*4));
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      int alu1 = (gidx1+(ridx1*4));
      int alu2 = (alu1%5);
      int alu3 = (alu0+(alu1/10));
      int alu4 = (alu3%5);
      float val0 = (((alu1<10)*(alu0<10)*(alu2<3)*(alu4<3)))?(*(data1+(alu2*4)+((alu1/5)%2)+(alu4*12)+(((alu3/5)%2)*2))):0.0f;
      acc0 = (val0+acc0);
    }
  }
  *(data0+(gidx0*4)+gidx1) = acc0;
}