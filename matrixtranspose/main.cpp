#include "kernels.cuh"
#include "helpers.cuh"

#include <stdio.h>
#include <chrono>

using Time = std::chrono::high_resolution_clock;
using mintime = std::nano;

#define WARMUP 20
#define REP 30

#define looprun(count)                                                   \
  for (size_t i = 0; i < count; i++)                                     \
  {                                                                      \
    trasnspose_V1 <<< blks, thpblk >>> (N, M1.getDevPtr(), M2.getDevPtr()); \
  }

int main(int argc, char const *argv[])
{
  size_t N = 0;

  if (argc != 2)
    N = 1 << 14;
  else
    N = 1 << atoi(argv[1]);

  auto [blks, thpblk] = getGpuLaunchConfig(N);
  getLasterror;
  host_dev_arr<double> M1(N * N, true);
  host_dev_arr<double> M2(N * N);
  M1.toDevice();

  size_t matrisizeinbytes = M1.getsizeinbytes();

  // warmup iterations
  looprun(WARMUP)

      cudaEvent_t start,
      stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto kernelstart = Time::now();
  looprun(REP);
  cudaDeviceSynchronize();
  auto kernelend = Time::now();

  double kerneltime = std::chrono::duration<double, mintime>(kernelend - kernelstart).count() / mintime::den;

  double arraydatasizeinGbytes = static_cast<double>(M1.getsizeinbytes()) / std::giga::num;

  printf("bandwidth (GB/s) : %4.5lf \n", arraydatasizeinGbytes * 2 * REP / kerneltime);
  M2.toHost();
  check_matrix_transpose(N, M1.getHostPtr(), M2.getHostPtr());

  return 0;
}
