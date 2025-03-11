#pragma once

#include <stdio.h>
#include <omp.h>

#define CUDA_ERR_CHK(CALL)                                    \
  {                                                           \
    auto error = CALL;                                        \
    if (error != cudaSuccess)                                 \
    {                                                         \
      fprintf(stderr,                                         \
              "GPUassert: %s %s %d\n",                        \
              cudaGetErrorString(error), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  }

#define CUDA_LAST_ERR(MSG)                                         \
  {                                                                \
    auto error = cudaGetLastError();                               \
    if (error != cudaSuccess)                                      \
    {                                                              \
      fprintf(stderr,                                              \
              "GPUassert: %s %s %s %d\n",                          \
              MSG, cudaGetErrorString(error), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  }

template <typename VT>
VT *devicemalloc(const size_t numelems)
{
  VT *ptr = nullptr;
  CUDA_ERR_CHK(cudaMalloc(&ptr, numelems * sizeof(VT)));
  return ptr;
}

template <typename VT>
VT *hostmalloc(const size_t numelems)
{
  VT *ptr = nullptr;
  CUDA_ERR_CHK(cudaMallocHost(&ptr, numelems * sizeof(VT)));
  return ptr;
}

template <typename VT>
void cpyhost2dev(VT *host, VT *dev, const size_t numelems)
{
  CUDA_ERR_CHK(cudaMemcpy(dev, host, numelems * sizeof(VT), cudaMemcpyHostToDevice));
}

template <typename VT>
void cpydev2host(VT *dev, VT *host, const size_t numelems)
{
  CUDA_ERR_CHK(cudaMemcpy(host, dev, numelems * sizeof(VT), cudaMemcpyDeviceToHost));
}

template <typename VT>
size_t sizeinbytes(size_t numelems)
{
  return sizeof(VT) * numelems;
}

void dummy_threadfunc()
{
  int numthreads = 1;
#pragma omp parallel
#pragma omp single
  numthreads = omp_get_num_threads();

  std::cout << "number of threads used " << numthreads << std::endl;
#ifdef _OPENMP
  std::cout << "number of threads used " << numthreads << std::endl;
#endif
}