#include "kernels.cuh"
#include "helpers.cuh"

#include <stdio.h>

int main(int argc, char const *argv[])
{
  size_t N = 5;

  auto [blks, thpblk] = getGpuLaunchConfig(N);
  getLasterror;

  host_dev_arr<double> M1(N * N, true);
  host_dev_arr<double> M2(N * N);
  host_dev_arr<double> M3(N * N);
  M1.toDevice();
  double *M2_hos = M2.getHostPtr();
  double *M2_dev = M2.getDevPtr();

  #if 0
  double *M1_hos = M1.getHostPtr();
  double *M1_dev = M1.getDevPtr();
  double *M3_hos = M3.getHostPtr();
  double *M3_dev = M3.getDevPtr();
  trasnspose_V0<<<blks, thpblk>>>(N, M1_dev, M2_dev);
  printMatrix(N, M1_hos);
  trasnspose_cpu(N, M1_hos, M3_hos);
  
  M2.toHost();
  printMatrix(N, M2_hos);
  printMatrix(N, M3_hos);
  check_matrix_equality(N, M3_hos, M2_hos);
#endif

  matrix_fillinindex<<<blks, thpblk>>>(N, M2_dev);
  getLasterror;
  M2.toHost();
  
  printMatrix(N, M2_hos);

  return 0;
}