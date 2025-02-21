#include <iostream>
#include <thrust/reduce.h>

int main(int argc, char const *argv[])
{
  size_t n_ele = 1 << 10;
  if (argc == 2)
  {
    n_ele = 9 * 1 << atoll(argv[1]);
  }

  size_t ele_size_bytes = n_ele * sizeof(double);

  double *host, *device;

  cudaMallocHost(&host, ele_size_bytes);

  thrust::fill(host, host + n_ele, 1.0);

  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < n_ele; i++)
  {
    sum += host[i];
  }

  std::cout << sum << std::endl;

  return 0;
}
