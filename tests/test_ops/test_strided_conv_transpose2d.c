#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_4_9_7_4_3_3n1(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 2 */
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.x; /* 63 */
  float acc0 = 0.0f;
  int alu0 = (gidx2%7);
  int alu1 = (gidx2/7);
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      int alu2 = (alu1+ridx1);
      for (int ridx2 = 0; ridx2 < 3; ridx2++) {
        int alu3 = (alu0+ridx2);
        int alu4 = (alu3+3);
        int alu5 = (alu2+(alu4/5)+1);
        int alu6 = (ridx2*(-1));
        float val0 = (((((alu0*(-1))+alu6)<(-1))*(alu3<7)*(((alu1*(-1))+(ridx1*(-1)))<(-1))*(alu2<9)*((alu5%2)<1)))?(*(data1+(alu4%5)+((((gidx0*16)+(ridx0*4)+(alu5/2)+30)%32)*5))):0.0f;
        float val1 = *(data2+(gidx1*9)+(ridx0*36)+(ridx1*(-3))+alu6+8);
        acc0 = ((val0*val1)+acc0);
      }
    }
  }
  float alu7 = max(acc0,0.0f);
  *(data0+(gidx0*252)+(gidx1*63)+(alu1*7)+alu0) = alu7;
}