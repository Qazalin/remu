#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_256_16_16_64(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 256 */
  int gidx1 = blockIdx.y; /* 16 */
  int gidx2 = blockIdx.x; /* 16 */
  float acc0 = 0.0f;
  int alu0 = (gidx0*1024);
  for (int ridx0 = 0; ridx0 < 64; ridx0++) {
    float val0 = *(data1+alu0+(gidx1*64)+ridx0);
    float val1 = *(data2+alu0+(gidx2*64)+ridx0);
    acc0 = ((val0*val1)+acc0);
  }
  *(data0+(gidx0*256)+(gidx1*16)+gidx2) = (acc0*0.125f);
}