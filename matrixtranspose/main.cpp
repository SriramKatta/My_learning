#include "kernels.cuh"
#include "helpers.cuh"

#include <stdio.h>

#define WARMUP 20
#define REP 30

#define looprun(count)                                                  \
  for (size_t i = 0; i < count; i++)                                    \
  {                                                                     \
    trasnspose_V0<<<blks, thpblk>>>(N, M1.getDevPtr(), M2.getDevPtr()); \
  }

int main(int argc, char const *argv[])
{
  size_t N = 0;

  if (argc != 2)
    N = 1 << 13;
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

  cudaEventRecord(start, 0);
  looprun(REP);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  M1.toHost();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float sec = (milliseconds * 1e3) / REP;
  double GB = static_cast<double>(M1.getsizeinbytes()) / 1e9;
  printf("Effective Bandwidth (GB/s): %4.5lf\n",  GB/ sec);
  return 0;
}
