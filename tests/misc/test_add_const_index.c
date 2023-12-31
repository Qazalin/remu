#include <hip/hip_runtime.h>

extern "C" __global__ void E_2(float* data0, const float* data1, const float* data2) {
  int gidx0 = 0x1;
  float val0 = *(data1 + gidx0);
  *(data0 + gidx0) = (val0 + val0);
}
