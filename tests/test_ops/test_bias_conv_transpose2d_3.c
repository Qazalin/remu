#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_4_121_9_4(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 2 */
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.x; /* 1089 */
  float acc0 = 0.0f;
  int alu0 = (gidx2%9);
  int alu1 = (gidx2/9);
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val0 = *(data1+(gidx1*36)+(alu0*(-1))+(ridx0*9)+8);
    float val1 = *(data2+(gidx0*484)+alu1+(ridx0*121));
    acc0 = ((val0*val1)+acc0);
  }
  *(data0+(gidx0*4356)+(gidx1*1089)+(alu1*9)+alu0) = acc0;
}