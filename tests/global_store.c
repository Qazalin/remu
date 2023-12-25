#include <hip/hip_common.h>
extern "C" __global__ void __launch_bounds__ (1, 1) E_4(int* data0) {
  *(data0+0) = 42;
}
