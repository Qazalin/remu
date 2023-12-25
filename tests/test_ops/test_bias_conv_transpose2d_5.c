#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_4_3_3_2_11_11(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 4 */
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.x; /* 9 */
  float acc0 = 0.0f;
  int alu0 = (gidx2%3);
  int alu1 = (gidx2/3);
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    for (int ridx1 = 0; ridx1 < 11; ridx1++) {
      for (int ridx2 = 0; ridx2 < 11; ridx2++) {
        int alu2 = (alu0+ridx2);
        float val0 = (((((alu0*(-1))+(ridx2*(-1)))<(-1))*(alu2<11)*(((alu1*(-1))+(ridx1*(-1)))<(-1))*((alu1+ridx1)<11)))?(*(data1+alu2+(alu1*9)+(ridx1*9)+(gidx1*81)+(ridx0*324)+(-20))):0.0f;
        float val1 = *(data2+(gidx0*121)+(ridx0*484)+(ridx1*11)+ridx2);
        acc0 = ((val0*val1)+acc0);
      }
    }
  }
  *(data0+(gidx0*36)+(gidx1*9)+(alu1*3)+alu0) = acc0;
}