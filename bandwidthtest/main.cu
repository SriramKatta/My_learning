#include <stdio.h>

#define WARMPUP 10
#define REP 100

#define pri(arg) printf(#arg);

template <typename VT>
__global__ 
void copyBandwidth(const VT *__restrict_arr data, size_t N)
{
  auto gridStart = threadIdx.x + blockDim.x * blockIdx.x;
  auto gridStride = blockDim.x * gridDim.x;
  VT val = 0;
  for (size_t i = gridStart; i < N; i += gridStride)
  {
    val = data[i];
    if (val < -1000.0)
      return;
  }
}
template<typename VT>
void launchKernel(void (*kernel)(const VT*, size_t), VT* data, size_t N , int blocks, int threads, cudaStream_t stream=0){
  int smemsize = threads * sizeof(VT);
  kernel<<<blocks, threads, smemsize, stream>>>(data, N);
}

int main()
{
  pri(__restrict_arr);
  return 0;
  double *data;
  size_t N = 1 << 20;
  size_t sizeBytes = N * sizeof(*data);
  cudaMallocManaged(&data, sizeBytes );
  cudaMemPrefetchAsync(data, sizeBytes, cudaCpuDeviceId);
  int devID;
  cudaGetDevice(&devID);
  int ws, numsms;
  cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, devID);
  cudaDeviceGetAttribute(&numsms, cudaDevAttrMultiProcessorCount, devID);

  int threads = ws * 16;
  int blks = numsms * 40;

  for (size_t i = 0; i < N; i++)
  {
    data[i] = 1.0;
  }
  cudaMemPrefetchAsync(data, sizeBytes, devID);
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (size_t i = 0; i < WARMPUP; i++)
  {
    launchKernel(&copyBandwidth, data, N, blks, threads, stream1);
  }
  


  cudaEventRecord(start, stream1);
  for (size_t i = 0; i < REP; i++)
  {
    launchKernel(&copyBandwidth, data, N, blks, threads, stream1);
  }

  cudaEventRecord(stop, stream1);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("band width : %f \n", (sizeBytes * 1e-9)/ (milliseconds * 1e-3));


  


  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream1);
  cudaFree(data);
return 0;
}
