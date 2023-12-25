#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) r_2_2_2n10(unsigned short* data0, const unsigned short* data1) {
  int gidx0 = blockIdx.y; /* 2 */
  int gidx1 = blockIdx.x; /* 2 */
  unsigned int acc0 = (unsigned int)(0);
  int alu0 = (gidx0*2);
  for (int ridx0 = 0; ridx0 < 2; ridx0++) {
    unsigned short val0 = *(data1+alu0+ridx0);
    acc0 = ((val0*((((gidx1+(ridx0*2))%3)<1)?(unsigned short)(1):(unsigned short)(0)))+acc0);
  }
  *(data0+alu0+gidx1) = (unsigned short)(acc0);
}