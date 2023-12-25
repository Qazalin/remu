#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_4_13_13_4_4(float* data0, const float* data1) {
  int gidx0 = blockIdx.z; /* 2 */
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.x; /* 169 */
  float acc0 = 0.0f;
  int alu0 = (gidx2%13);
  int alu1 = (gidx2/13);
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu2 = (alu1+(ridx0*13));
    for (int ridx1 = 0; ridx1 < 4; ridx1++) {
      int alu3 = (alu0+(ridx1*13));
      int alu4 = (alu3%14);
      int alu5 = (alu2+(alu3/42));
      int alu6 = (alu5%14);
      float val0 = (((alu3<42)*(alu2<42)*(alu4<11)*(alu6<11)))?(*(data1+(alu4*9)+((alu3/14)%3)+(alu6*99)+(((alu5/14)%3)*3)+((((gidx0*4)+gidx1+(alu5/42))%8)*1089))):0.0f;
      acc0 = (val0+acc0);
    }
  }
  *(data0+(gidx0*676)+(gidx1*169)+(alu1*13)+alu0) = acc0;
}