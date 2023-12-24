#include <hip/hip_common.h>
extern "C" __global__ void __launch_bounds__ (1, 1) E_4(int* data0, const int* data1, const int* data2) {
  int gidx0 = blockIdx.x; /* 4 */
  int val0 = *(data1+gidx0);
  int val1 = *(data2+gidx0);
  *(data0+gidx0) = (val0+val1);
}
