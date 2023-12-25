#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_11_28_3_3_3(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 4 */
  int gidx1 = blockIdx.y; /* 11 */
  int gidx2 = blockIdx.x; /* 28 */
  float acc0 = 0.0f;
  int alu0 = (gidx1*28);
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      for (int ridx2 = 0; ridx2 < 3; ridx2++) {
        int alu1 = (gidx2+ridx2);
        float val0 = (((((gidx2*(-1))+(ridx2*(-1)))<0)*(alu1<29)*(((gidx1*(-1))+(ridx1*(-1)))<0)*((gidx1+ridx1)<12)))?(*(data1+alu1+alu0+(ridx1*28)+(ridx0*308)+(-29))):0.0f;
        float val1 = *(data2+(gidx0*27)+(ridx0*9)+(ridx1*3)+ridx2);
        acc0 = ((val0*val1)+acc0);
      }
    }
  }
  float alu2 = max(acc0,0.0f);
  *(data0+(gidx0*308)+alu0+gidx2) = alu2;
}