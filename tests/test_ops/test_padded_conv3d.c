#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_9_9_9_4_3_3_3(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 4 */
  int gidx1 = blockIdx.y; /* 9 */
  int gidx2 = blockIdx.x; /* 81 */
  float acc0 = 0.0f;
  int alu0 = (gidx2%9);
  int alu1 = (gidx2/9);
  int alu2 = (alu1*9);
  int alu3 = (gidx1*81);
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      for (int ridx2 = 0; ridx2 < 3; ridx2++) {
        for (int ridx3 = 0; ridx3 < 3; ridx3++) {
          int alu4 = (alu0+ridx3);
          float val0 = (((((alu0*(-1))+(ridx3*(-1)))<0)*(alu4<10)*(((alu1*(-1))+(ridx2*(-1)))<0)*((alu1+ridx2)<10)*(((gidx1*(-1))+(ridx1*(-1)))<0)*((gidx1+ridx1)<10)))?(*(data1+alu4+alu2+(ridx2*9)+alu3+(ridx1*81)+(ridx0*729)+(-91))):0.0f;
          float val1 = *(data2+(gidx0*108)+(ridx0*27)+(ridx1*9)+(ridx2*3)+ridx3);
          acc0 = ((val0*val1)+acc0);
        }
      }
    }
  }
  float alu5 = max(acc0,0.0f);
  *(data0+(gidx0*729)+alu3+alu2+alu0) = alu5;
}