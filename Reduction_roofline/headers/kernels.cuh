#pragma once
#include "helpers.cuh"
#include <omp.h>

template <typename VT>
void fill_data(VT *ptr, size_t n_ele)
{
#pragma omp parallel for
  for (size_t i = 0; i < n_ele; i++)
  {
    ptr[i] = 1.0;
  }
}

template <typename VT>
VT sum_cpu(VT *ptr, size_t n_ele)
{
  auto sum = static_cast<VT>(0);
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < n_ele; i++)
  {
    sum += ptr[i];
  }
  return sum;
}

template <typename VT>
__global__ void sum_gpu_kernel(VT *ptr, size_t n_ele, VT *ans)
{
  extern __shared__ VT sharedmem[];
  VT * s_data = reinterpret_cast<VT *>(sharedmem);

  const auto tid = threadIdx.x;

  s_data[tid] = static_cast<VT>(0);

  const auto gridStart = threadIdx.x + blockDim.x * blockIdx.x;
  const auto gridStride = gridDim.x * blockDim.x;

  for (size_t idx = gridStart; idx < n_ele; idx += gridStride)
    s_data[tid] += ptr[idx];

  for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
  {
    __syncthreads();
    if (tid < s) 
      s_data[tid] += s_data[tid + s];
  }

  if (tid == 0) atomicAdd(ans, s_data[0]);
}



template <typename VT>
VT sum_gpu(VT *h_arr, size_t n_ele)
{
  auto *d_arr = devicemalloc<VT>(n_ele);
  auto *d_res = devicemalloc<VT>(1);
  cpyhost2dev(h_arr, d_arr, n_ele);
  cudaMemset(d_res, 0, sizeof(VT));
  int blks = 32;
  int thpblk = 512;
  sum_gpu_kernel<<<blks, thpblk, thpblk * sizeof(VT), 0>>>(d_arr, n_ele, d_res);
  CUDA_LAST_ERR("GPU LAUNCH FAIL")
  VT h_res = 0;
  cpydev2host(d_res, &h_res, 1);
  CUDA_ERR_CHK(cudaFree(d_arr));
  CUDA_ERR_CHK(cudaFree(d_res));
  return h_res;
}