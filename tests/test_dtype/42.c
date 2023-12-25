#include <hip/hip_fp16.h>

typedef union { struct { half x, y, z, w; } __attribute__((aligned(8))); half data[4]; } half4;
__device__ half4 make_half4(half x, half y, half z, half w) { return {x, y, z, w}; }
typedef union { struct { half x, y, z, w, a, b, c, d; } __attribute__((aligned(16))); half data[8]; } half8;
__device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { return {x, y, z, w, a, b, c, d}; }
 typedef _Float16 half16 __attribute__((ext_vector_type(16)));
__device__ half16 make_half16(half x, half y, half z, half w, half a, half b, half c, half d,
                              half e, half f, half g, half h, half i, half j, half k, half l) {
                                return {x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; }
  
#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_2_2n4(half* data0) {
  int gidx0 = blockIdx.y; /* 2 */
  int gidx1 = blockIdx.x; /* 2 */
  *(data0+(gidx0*2)+gidx1) = (half)(((((gidx0+(gidx1*2))%3)<1)?(long)(1):(long)(0)));
}