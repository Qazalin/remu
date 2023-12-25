#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_4_15_32_3(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 4 */
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.x; /* 480 */
  float acc0 = 0.0f;
  int alu0 = (gidx2/32);
  int alu1 = (gidx2%32);
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    float val0 = ((((alu1*(-1))<(-1))*(alu1<30)*((alu0*(-1))<(-1))*(alu0<13)))?(*(data1+(gidx0*924)+(alu0*28)+alu1+(ridx0*308)+(-58))):0.0f;
    float val1 = *(data2+(gidx1*3)+ridx0);
    acc0 = ((val0*val1)+acc0);
  }
  float alu2 = max(acc0,0.0f);
  *(data0+(gidx0*1920)+(gidx1*480)+(alu0*32)+alu1) = alu2;
}