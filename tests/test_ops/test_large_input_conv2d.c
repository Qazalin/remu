#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_4_6_60_63_16_5_2(float* data0, const float* data1, const float* data2) {
  int gidx0 = blockIdx.z; /* 4 */
  int gidx1 = blockIdx.y; /* 6 */
  int gidx2 = blockIdx.x; /* 3780 */
  float acc0 = 0.0f;
  int alu0 = (gidx2/63);
  int alu1 = (gidx2%63);
  for (int ridx0 = 0; ridx0 < 16; ridx0++) {
    for (int ridx1 = 0; ridx1 < 5; ridx1++) {
      for (int ridx2 = 0; ridx2 < 2; ridx2++) {
        float val0 = *(data1+(gidx0*65536)+(alu0*64)+alu1+(ridx0*4096)+(ridx1*64)+ridx2);
        float val1 = *(data2+(gidx1*160)+(ridx0*10)+(ridx1*2)+ridx2);
        acc0 = ((val0*val1)+acc0);
      }
    }
  }
  float alu2 = max(acc0,0.0f);
  *(data0+(gidx0*22680)+(gidx1*3780)+(alu0*63)+alu1) = alu2;
}