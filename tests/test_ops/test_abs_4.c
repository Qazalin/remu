#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_n2(float* data0, const float* data1) {
  float val0 = *(data1+0);
  float alu0 = max((-0.1464405059814453f),0.0f);
  float alu1 = (val0+1.0f);
  float alu2 = (alu1+alu1);
  float alu3 = max(0.1464405059814453f,0.0f);
  *(data0+0) = ((-((float)((0.0f<alu0))*alu2))+((float)((0.0f<alu3))*alu2));
}