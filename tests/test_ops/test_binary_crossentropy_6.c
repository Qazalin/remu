#include <hip/hip_common.h>
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
  typedef float float8 __attribute__((ext_vector_type(8)));
  __device__ float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return {x, y, z, w, a, b, c, d}; }
  extern "C" __global__
  void __launch_bounds__ (1, 1) E_320n3(float* data0, const float* data1, const float* data2, const float* data3) {
  int gidx0 = blockIdx.x; /* 320 */
  float val0 = *(data1+gidx0);
  float val1 = *(data2+gidx0);
  float val2 = *(data3+0);
  bool alu0 = (0.0f<val0);
  bool alu1 = (val0<0.0f);
  float alu2 = (-(alu1?0.0f:(alu0?val0:(val0*0.5f))));
  bool alu3 = ((-1.0f)<alu2);
  float alu4 = ((alu2<(-1.0f))?0.0f:(-(val1*(-val2))));
  float alu5 = (alu1?0.0f:(-((alu3?alu4:0.0f)+(0.5f*(alu3?0.0f:alu4)))));
  *(data0+gidx0) = ((alu0?alu5:0.0f)+(0.5f*(alu0?0.0f:alu5)));
}