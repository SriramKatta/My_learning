#include <iostream>
#include <thrust/reduce.h>

void fill_data(double *ptr, size_t n_ele)
{
#pragma omp parallel for
  for (size_t i = 0; i < n_ele; i++)
  {
    ptr[i] = 1.0;
  }
}

long double sum_cpu(double *ptr, size_t n_ele)
{
  long double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < n_ele; i++)
  {
    sum += ptr[i];
  }
  return sum;
}

int main(int argc, char const *argv[])
{
  size_t n_ele = 1 << 10;
  if (argc == 2)
  {
    n_ele = 1 << atoll(argv[1]);
  }

  size_t ele_size_bytes = n_ele * sizeof(double);

  double *host, *device;

  auto error = cudaMallocHost(&host, ele_size_bytes);
  error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
  }

  long double sum;
  for (size_t i = 0; i < 100; i++)
  {
    /* code */
    sum = sum_cpu(host, n_ele);
  }

  std::cout << sum << std::endl;

  return 0;
}
