#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_6_6_3_3(float* data0, const float* data1) {
  int gidx0 = blockIdx.y; /* 6 */
  int gidx1 = blockIdx.x; /* 6 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    int alu0 = (gidx0+(ridx0*6));
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      int alu1 = (gidx1+(ridx1*6));
      int alu2 = (alu1%7);
      int alu3 = (alu0+(alu1/14));
      int alu4 = (alu3%7);
      float val0 = (((alu1<14)*(alu0<14)*(alu2<5)*(alu4<5)))?(*(data1+(alu2*4)+((alu1/7)%2)+(alu4*20)+(((alu3/7)%2)*2))):0.0f;
      acc0 = (val0+acc0);
    }
  }
  *(data0+(gidx0*6)+gidx1) = acc0;
}