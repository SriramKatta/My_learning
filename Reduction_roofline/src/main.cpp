#include <iostream>
#include "kernels.cuh"
#include "helpers.cuh"

#define DT int
#define REP 10

int main(int argc, char const *argv[])
{
  dummy_threadfunc();
  size_t n_ele = 1 << 30;
  size_t ele_size_bytes = sizeinbytes<DT>(n_ele);
  auto *h_arr = hostmalloc<DT>(n_ele);
  fill_data(h_arr, n_ele);
  auto gpu_ans = static_cast<DT>(0);

  for (size_t i = 0; i < REP; i++)
    gpu_ans = sum_gpu(h_arr, n_ele);

  std::cout << "sum on gpu is : " << gpu_ans << std::endl;

  auto cpu_ans = static_cast<DT>(0);

  for (size_t i = 0; i < REP; i++)
    cpu_ans = sum_cpu(h_arr, n_ele);

  std::cout << "sum on cpu is : " << cpu_ans << std::endl;
  CUDA_ERR_CHK(cudaFreeHost(h_arr));
  return 0;
}
